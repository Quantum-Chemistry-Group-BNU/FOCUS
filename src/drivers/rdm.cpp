#include <iostream>
#include <iomanip>
#include <string>
#include "core/mem_status.h"
#include "core/perfcomm.h"
#include "core/integral_io.h"
#include "io/io.h"
#include "io/input.h"
#include "ci/ci_header.h"
#include "ctns/ctns_header.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GPU
#include "gpu/gpu_env.h"
#endif

using namespace std;
using namespace fock;

template <typename Qm, typename Tm>  
void RDM(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in RDM!");
   }

   //-----------------------------------------------------------------
   // The driver RDM only support MPS with the same kind (topo,Qm,Tm)
   //-----------------------------------------------------------------

   // initialization of MPS and save to binary format
   bool exist1 = false;
   ctns::comb<Qm,Tm> icomb;
   if(schd.ctns.rcanon_file.size()>0){
      ctns::comb_load(icomb, schd, schd.ctns.rcanon_file);
      assert(icomb.topo.ifmps);
      if(rank == 0){
         icomb.display_shape();
         icomb.display_size();
         if(schd.ctns.savebin) ctns::rcanon_savebin(icomb, schd.scratch+"/"+schd.ctns.rcanon_file);
      }
      exist1 = true;
   }
   if(schd.ctns.rcanon_file.size() == 0){
      tools::exit("error: MPS1 must be given!");
   }
   bool is_same = false;
   ctns::comb<Qm,Tm> icomb2;
   if(schd.ctns.rcanon2_file.size()>0){
      ctns::comb_load(icomb2, schd, schd.ctns.rcanon2_file);
      assert(icomb2.topo.ifmps); // must be MPS
      if(rank == 0){
         icomb2.display_shape();
         icomb2.display_size();
         if(schd.ctns.savebin) ctns::rcanon_savebin(icomb2, schd.scratch+"/"+schd.ctns.rcanon2_file);
      }
   }else{
      auto jcomb = icomb;
      icomb2 = std::move(jcomb);
      is_same = (schd.ctns.iroot == schd.ctns.jroot);
   }
   const int iroot = schd.ctns.iroot;
   const int jroot = schd.ctns.jroot;

   // display task_prop
   assert(schd.sorb == 2*icomb.get_nphysical());
   const size_t k = schd.sorb;
   const size_t k2 = k*(k-1)/2;
   const size_t k3 = k*(k-1)*(k-2)/6;
   if(rank == 0){
      ctns::rdm_checkinput(schd, icomb, icomb2, is_same);
   } // rank-0

   //---------
   // overlap
   //---------
   if(tools::is_in_vector(schd.ctns.task_prop,"ova")){
      if(rank == 0){
         auto Sij = get_Smat(icomb, icomb2);
         std::cout << std::endl;
         Sij.print("<MPS1|MPS2>", schd.ctns.outprec);
      }
   } 

   // --- rdms ---
   const double thresh = 1.e-6;
   ctns::rdmaux<Tm> aux;
   linalg::matrix<Tm> rdm1;
   linalg::matrix<Tm> rdm2;
   linalg::matrix<Tm> rdm3;

   //------
   // rdm1 
   //------
   if(tools::is_in_vector(schd.ctns.task_prop,"1p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdm1tmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdm1tmp = ctns::rdm1_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdm1tmp, 0);
#endif
         aux.rdm = rdm1tmp; 
      } // debug

      // compute rdm1 
      rdm1.resize(k,k);
      ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian and rank == 0){
         auto diff = rdm1 - rdm1tmp;
         std::cout << "\ndebug_rdm[1p1h]: |rdm1|=" << rdm1.normF()
            << " |rdm1tmp|=" << rdm1tmp.normF()
            << " diff|rdm1-rdm1tmp|=" << diff.normF() 
            << std::endl;
         assert(diff.normF() < thresh);
      } // debug
   } // rdm1

   //------
   // rdm2
   //------
   if(tools::is_in_vector(schd.ctns.task_prop,"2p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdm2tmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdm2tmp = ctns::rdm2_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdm2tmp, 0);
#endif
         aux.rdm = rdm2tmp;
      }      

      // compute rdm2
      rdm2.resize(k2,k2);
      ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2, aux);

      if(rank == 0){
         if(schd.ctns.debug_rdm and Qm::ifabelian){
            auto diff = rdm2 - rdm2tmp;
            std::cout << "\ndebug_rdm[2p2h]: |rdm2|=" << rdm2.normF()
               << " |rdm2tmp|=" << rdm2tmp.normF()
               << " diff|rdm2-rdm2tmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } // debug

         // hamiltonian matrix elements if needed 
         integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         integral::load(int2e, int1e, ecore, schd.integral_file);
         assert(schd.sorb == int1e.sorb);
         int nelec2 = icomb2.get_qsym_state().ne();
         rdm1 = get_rdm1_from_rdm2(rdm2, false, nelec2);
         auto Sij = rdm1.trace()/Tm(schd.nelec);
         auto Hij = get_etot(rdm2, rdm1, int2e, int1e) + Sij*ecore; 
         std::cout << "iroot=" << iroot << " jroot=" << jroot
            << " H(i,j)=" << std::fixed << std::setprecision(schd.ctns.outprec) << Hij 
            << std::endl;
         fock::get_cumulant(rdm2, rdm1);

         // single-site entropy analysis: obtainable from 2-RDM
         if(is_same){
            auto s1 = fock::entropy1site(rdm2, rdm1);
            if(schd.ctns.debug_rdm and Qm::ifabelian){
               auto s1tmp = ctns::entropy1_simple(icomb, iroot);
               linalg::xaxpy(s1.size(), -1.0, s1.data(), s1tmp.data());
               double diff = linalg::xnrm2(s1tmp.size(), s1tmp.data());
               assert(diff < thresh);
            }
         }
      } // rank-0
   } // rdm2

   //----------
   // tdm-1p0h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"1p0h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdmtmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdmtmp = ctns::tdm1p0h_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
         aux.rdm = rdmtmp;
      }      

      linalg::matrix<Tm> rdm(k,1);
      ctns::rdm_sweep("1p0h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            auto diff = rdm - rdmtmp;
            std::cout << "\ndebug_rdm[1p0h]: |rdm|=" << rdm.normF()
               << " |rdmtmp|=" << rdmtmp.normF()
               << " diff|rdm-rdmtmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } // debug

         // debug via computing rdm1
         if(schd.ctns.alg_rdm > 0){
            linalg::matrix<Tm> rdm1tmp(k,k), rdm1tmp2(k,k);
            // <psi|i+j|psi2> = <psi|i+(j|psi2>)
            ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1tmp, aux);
            auto image1 = icomb2.topo.get_image1();
            for(int j=0; j<k; j++){
               linalg::matrix<Tm> tdm(k,1);
               int kj = j/2, spin_j = j%2;
               auto icomb2_j = apply_opC(icomb2, kj, spin_j, 0); // j|psi2>
               ctns::rdm_sweep("1p0h", false, icomb, icomb2_j, schd, scratch, tdm, aux);
               int pj = 2*image1[kj] + spin_j; // map to the orbital index
               linalg::xcopy(k, tdm.data(), rdm1tmp2.col(pj));
            }
            if(rank == 0){
               auto rdm1diff = rdm1tmp2 - rdm1tmp;
               std::cout << "\ndebug_rdm: 1p0h via rdm1: |rdm1|=" << rdm1tmp.normF() 
                  << " |rdm1b|=" << rdm1tmp2.normF()
                  << " |rdm1-rdm1b|=" << rdm1diff.normF() 
                  << std::endl;
               assert(rdm1diff.normF() < thresh);
            }
         }
      }
   } // tdm-1p0h

   //----------
   // tdm-0h1h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"0p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdm(1,k);
      ctns::rdm_sweep("0p1h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         // debug via computing rdm1
         if(schd.ctns.alg_rdm > 0){
            linalg::matrix<Tm> rdm1tmp(k,k), rdm1tmp2(k,k);
            // <psi|i+j|psi2> = (<psi|i+)j|psi2>
            ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1tmp, aux);
            auto image1 = icomb.topo.get_image1();
            for(int i=0; i<k; i++){
               linalg::matrix<Tm> tdm(1,k); 
               int ki = i/2, spin_i = i%2;
               auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi>
               ctns::rdm_sweep("0p1h", false, icomb_i, icomb2, schd, scratch, tdm, aux);
               int pi = 2*image1[ki] + spin_i; // map to the orbital index 
               linalg::xcopy(k, tdm.data(), 1, rdm1tmp2.row(pi), k);
            }
            if(rank == 0){
               auto rdm1diff = rdm1tmp2 - rdm1tmp;
               std::cout << "\ndebug_rdm: 0p1h via rdm1: |rdm1|=" << rdm1tmp.normF() 
                  << " |rdm1b|=" << rdm1tmp2.normF()
                  << " |rdm1-rdm1b|=" << rdm1diff.normF() 
                  << std::endl;
               assert(rdm1diff.normF() < thresh);
            }
         }
      }
   } // tdm-0p1h

   //----------
   // tdm-2p0h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"2p0h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdmtmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdmtmp = ctns::tdm2p0h_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
         aux.rdm = rdmtmp;
      }      

      linalg::matrix<Tm> rdm(k2,1);
      ctns::rdm_sweep("2p0h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            auto diff = rdm - rdmtmp;
            std::cout << "\ndebug_rdm[2p0h]: |rdm|=" << rdm.normF()
               << " |rdmtmp|=" << rdmtmp.normF()
               << " diff|rdm-rdmtmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } // debug

         // debug via computing rdm2
         if(schd.ctns.alg_rdm > 0){
            linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2);
            // <psi|i+j+rs|psi2> (r<s) = <psi|i+j+(rs|psi2>)
            ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
            auto image1 = icomb2.topo.get_image1();
            for(int s=0; s<k; s++){
               for(int r=0; r<s; r++){
                  linalg::matrix<Tm> tdm(k2,1);
                  int ks = s/2, spin_s = s%2;
                  int kr = r/2, spin_r = r%2;
                  auto icomb2_s = apply_opC(icomb2, ks, spin_s, 0); // s|psi2>
                  auto icomb2_rs = apply_opC(icomb2_s, kr, spin_r, 0); // rs|psi2>
                  ctns::rdm_sweep("2p0h", false, icomb, icomb2_rs, schd, scratch, tdm, aux);
                  int ps = 2*image1[ks] + spin_s; // map to the orbital index
                  int pr = 2*image1[kr] + spin_r;
                  auto psr = tools::canonical_pair0(ps,pr);
                  Tm sgn = tools::sgn_pair0(ps,pr);  
                  tdm *= sgn;
                  linalg::xcopy(k2, tdm.data(), rdm2tmp2.col(psr));
               }
            }
            if(rank == 0){
               auto rdm2diff = rdm2tmp2 - rdm2tmp;
               std::cout << "\ndebug_rdm: 2p0h via rdm2: |rdm2|=" << rdm2tmp.normF() 
                  << " |rdm2b|=" << rdm2tmp2.normF()
                  << " |rdm2-rdm2b|=" << rdm2diff.normF() 
                  << std::endl;
               assert(rdm2diff.normF() < thresh);
            }
         }
      }
   } // tdm-2p0h

   //----------
   // tdm-0p2h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"0p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdm(1,k2);
      ctns::rdm_sweep("0p2h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         // debug via computing rdm2
         if(schd.ctns.alg_rdm > 0){
            linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2);
            // <psi|i+j+rs|psi2> (i>j,r<s) = (<psi|i+j+)rs|psi2>, |tmp>=ji|psi> (i>j)
            ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
            auto image1 = icomb.topo.get_image1();
            for(int i=0; i<k; i++){
               for(int j=0; j<i; j++){
                  linalg::matrix<Tm> tdm(1,k2);
                  int ki = i/2, spin_i = i%2;
                  int kj = j/2, spin_j = j%2;
                  auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi2>
                  auto icomb_ji = apply_opC(icomb_i, kj, spin_j, 0); // ji|psi2>
                  ctns::rdm_sweep("0p2h", false, icomb_ji, icomb2, schd, scratch, tdm, aux);
                  int pi = 2*image1[ki] + spin_i; // map to the orbital index
                  int pj = 2*image1[kj] + spin_j;
                  auto pij = tools::canonical_pair0(pi,pj);
                  Tm sgn = tools::sgn_pair0(pi,pj);
                  tdm *= sgn;
                  linalg::xcopy(k2, tdm.data(), 1, rdm2tmp2.row(pij), k2);
               }
            }
            if(rank == 0){
               auto rdm2diff = rdm2tmp2 - rdm2tmp;
               std::cout << "\ndebug_rdm: 0p2h via rdm2: |rdm2|=" << rdm2tmp.normF() 
                  << " |rdm2b|=" << rdm2tmp2.normF()
                  << " |rdm2-rdm2b|=" << rdm2diff.normF() 
                  << std::endl;
               assert(rdm2diff.normF() < thresh);
            }
         }
      }
   } // tdm-0p2h

   //----------
   // tdm-2p1h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"2p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdmtmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdmtmp = ctns::tdm2p1h_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
         aux.rdm = rdmtmp;
      }      

      linalg::matrix<Tm> rdm(k2,k);
      ctns::rdm_sweep("2p1h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            auto diff = rdm - rdmtmp;
            std::cout << "\ndebug_rdm[2p1h]: |rdm|=" << rdm.normF()
               << " |rdmtmp|=" << rdmtmp.normF()
               << " diff|rdm-rdmtmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } // debug

         // debug via computing rdm2
         if(schd.ctns.alg_rdm > 0){
            linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2);
            // <psi|i+j+rs|psi2> (r<s) = <psi|i+j+r(s|psi2>)
            ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
            auto image1 = icomb2.topo.get_image1();
            for(int s=0; s<k; s++){
               linalg::matrix<Tm> tdm(k2,k);
               int ks = s/2, spin_s = s%2;
               auto icomb2_s = apply_opC(icomb2, ks, spin_s, 0); // s|psi2>
               ctns::rdm_sweep("2p1h", false, icomb, icomb2_s, schd, scratch, tdm, aux);
               int ps = 2*image1[ks] + spin_s; // map to the orbital index
               for(int pr=0; pr<ps; pr++){
                  auto psr = tools::canonical_pair0(ps,pr);
                  linalg::xcopy(k2, tdm.col(pr), rdm2tmp2.col(psr));
               }
            }
            if(rank == 0){
               auto rdm2diff = rdm2tmp2 - rdm2tmp;
               std::cout << "\ndebug_rdm: 2p1h via rdm2: |rdm2|=" << rdm2tmp.normF() 
                  << " |rdm2b|=" << rdm2tmp2.normF()
                  << " |rdm2-rdm2b|=" << rdm2diff.normF() 
                  << std::endl;
               assert(rdm2diff.normF() < thresh);
            }
         }
      }
   } // tdm-2p1h

   //----------
   // tdm-1p2h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"1p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdm(k,k2);
      ctns::rdm_sweep("1p2h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2);
         // <psi|i+j+rs|psi2> (i>j,r<s) = (<psi|i+)j+rs|psi2>, |tmp>=i|psi> 
         ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
         auto image1 = icomb.topo.get_image1();
         for(int i=0; i<k; i++){
            linalg::matrix<Tm> tdm(k,k2);
            int ki = i/2, spin_i = i%2;
            auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi2>
            ctns::rdm_sweep("1p2h", false, icomb_i, icomb2, schd, scratch, tdm, aux);
            int pi = 2*image1[ki] + spin_i; // map to the orbital index
            for(int pj=0; pj<pi; pj++){
               auto pij = tools::canonical_pair0(pi,pj);
               linalg::xcopy(k2, tdm.row(pj), k, rdm2tmp2.row(pij), k2);
            }
         }
         if(rank == 0){
            auto rdm2diff = rdm2tmp2 - rdm2tmp;
            std::cout << "\ndebug_rdm: 1p2h via rdm2: |rdm2|=" << rdm2tmp.normF() 
               << " |rdm2b|=" << rdm2tmp2.normF()
               << " |rdm2-rdm2b|=" << rdm2diff.normF() 
               << std::endl;
            assert(rdm2diff.normF() < thresh);
         }
      }
   } // tdm-1p2h

   //----------
   // tdm-3p2h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"3p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdmtmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdmtmp = ctns::tdm3p2h_simple(icomb, icomb2, iroot, jroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
         aux.rdm = rdmtmp;
      }      

      linalg::matrix<Tm> rdm(k3,k2);
      ctns::rdm_sweep("3p2h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            auto diff = rdm - rdmtmp;
            std::cout << "\ndebug_rdm[3p2h]: |rdm|=" << rdm.normF()
               << " |rdmtmp|=" << rdmtmp.normF()
               << " diff|rdm-rdmtmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } 
      } // debug
   } // tdm-3p2h

   //----------
   // tdm-2p3h
   //----------
   if(tools::is_in_vector(schd.ctns.task_prop,"2p3h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> rdmtmp;
      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            rdmtmp = ctns::tdm3p2h_simple(icomb2, icomb, jroot, iroot);
         }
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
         rdmtmp = rdmtmp.H(); // <I|O|J> = (<J|O^+|I>)^*
         aux.rdm = rdmtmp;
      }

      linalg::matrix<Tm> rdm(k2,k3);
      ctns::rdm_sweep("2p3h", false, icomb, icomb2, schd, scratch, rdm, aux);

      if(schd.ctns.debug_rdm and Qm::ifabelian){
         if(rank == 0){
            auto diff = rdm - rdmtmp;
            std::cout << "\ndebug_rdm[2p3h]: |rdm|=" << rdm.normF()
               << " |rdmtmp|=" << rdmtmp.normF()
               << " diff|rdm-rdmtmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         } 
      } // debug
   } // tdm-2p3h

   //------
   // rdm3
   //------
   if(tools::is_in_vector(schd.ctns.task_prop,"3p3h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      // compute rdm3 = <p+q+r+stu> (p>q>r,s<t<u)
      rdm3.resize(k3,k3);
      auto image1 = icomb.topo.get_image1();
      auto t0 = tools::get_time();
      for(int i=0; i<k; i++){
         auto tx = tools::get_time();
         int ki = i/2, spin_i = i%2;
         auto icomb2_i = apply_opC(icomb2, ki, spin_i, 0); // i|psi> (u=i)

         linalg::matrix<Tm> rdmtmp;
         if(schd.ctns.debug_rdm and Qm::ifabelian){
            if(rank == 0){
               rdmtmp = ctns::tdm3p2h_simple(icomb, icomb2_i, iroot, jroot);
            }
#ifndef SERIAL
            if(schd.ctns.debug_rdm and size > 1) mpi_wrapper::broadcast(icomb.world, rdmtmp, 0);
#endif
            aux.rdm = rdmtmp;
         }

         linalg::matrix<Tm> rdm32(k3,k2);
         ctns::rdm_sweep("3p2h", false, icomb, icomb2_i, schd, scratch, rdm32, aux);

         // copy data to rdm3 <Psi_0|p+q+r+st|Psi_i>
         int pi = 2*image1[ki] + spin_i; // map to the orbital index
         for(int pt=0; pt<pi; pt++){ 
            for(int ps=0; ps<pt; ps++){
               auto psti = tools::canonical_triple0(pi,pt,ps);
               auto pst = tools::canonical_pair0(pt,ps); 
               Tm sgn1 = tools::sgn_triple0(pi,pt,ps);
               Tm sgn2 = tools::sgn_pair0(pt,ps);
               linalg::xaxpy(k3, sgn1*sgn2, rdm32.col(pst), rdm3.col(psti)); 
            }
         }
         auto ty = tools::get_time();
         if(rank == 0) std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
      } 
      auto t1 = tools::get_time();
      if(rank == 0) std::cout << "total time for 3-RDM: " << tools::get_duration(t1-t0) << " S" << std::endl;

      // compared against other evaluation
      if(rank == 0){
         if(schd.ctns.debug_rdm and Qm::ifabelian){
            auto rdm3tmp = ctns::rdm3_simple(icomb, icomb2, iroot, jroot);
            auto diff = rdm3 - rdm3tmp;
            std::cout << "\ndebug_rdm[3p3h]: |rdm3|=" << rdm3.normF() 
               << " |rdm3tmp|=" << rdm3tmp.normF()
               << " diff|rdm3-rdm3tmp|=" << diff.normF() 
               << std::endl; 
            assert(diff.normF() < thresh);
         }
         // debug by computing energy
         int nelec2 = icomb2.get_qsym_state().ne();
         auto rdm2 = fock::get_rdm2_from_rdm3(rdm3, true, nelec2);
         auto rdm1 = fock::get_rdm1_from_rdm2(rdm2, true, nelec2);
         integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         integral::load(int2e, int1e, ecore, schd.integral_file);
         assert(schd.sorb == int1e.sorb);
         auto Sij = rdm1.trace()/Tm(schd.nelec);
         auto Hij = get_etot(rdm2, rdm1, int2e, int1e) + Sij*ecore; 
         std::cout << "iroot=" << iroot << " jroot=" << jroot
            << " H(i,j)=" << std::fixed << std::setprecision(schd.ctns.outprec) << Hij 
            << std::endl;
      } // rank-0
   } // rdm3

   //-----------
   // save rdms
   //-----------
   if(rank == 0 and is_same){
      if(rdm1.size()>0 or rdm2.size()>0 or rdm3.size()>0){
         std::cout << "\nsave results for rdms:" << std::endl;
      }
      // save text
      if(rdm1.size()>0) rdm1.save_txt(schd.scratch+"/rdm1mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm2.size()>0) rdm2.save_txt(schd.scratch+"/rdm2mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm3.size()>0) rdm3.save_txt(schd.scratch+"/rdm3mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      // compute natural orbitals
      if(rdm1.size()>0){
         auto natorbs = fock::get_natorbs(fock::get_rdm1s(rdm1));
         natorbs.save_txt(schd.scratch+"/natorbs", schd.ctns.outprec);
      }
   }

   //------------
   // dsrg-mrpt2
   //------------
   if(tools::is_in_vector(schd.ctns.task_prop,"mrpt2")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      // dsrg_mrpt2
      ctns::rdm_mrpt2(icomb, schd, scratch);
   } // mrpt2
}

int main(int argc, char *argv[]){
   int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
   // setup MPI environment 
   boost::mpi::environment env{argc, argv};
   boost::mpi::communicator world;
   rank = world.rank();
   size = world.size();
#endif
#ifdef _OPENMP
   maxthreads = omp_get_max_threads();
#endif
   if(rank == 0){
      tools::license();
      std::cout << "\nmpisize = " << size
         << " maxthreads = " << maxthreads
         << std::endl;
   }

   // read input
   string fname;
   if(argc == 1){
      tools::exit("error: no input file is given!");
   }else{
      fname = argv[1];
      if(rank == 0) cout << "\ninput file = " << fname << endl;
   }
   input::schedule schd;
   if(rank == 0) schd.read(fname);
#ifndef SERIAL
   if(size > 1){
      boost::mpi::broadcast(world, schd, 0);
      schd.world = world;
   }
#endif
   if(!schd.ctns.run){
      if(rank == 0) std::cout << "\ncheck input again, there is no task for RDM!" << std::endl;
      return 0;
   }

   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

   const bool ifgpu = schd.ctns.alg_renorm>10; // no need for considering alg_hvec
#ifdef GPU
   if(ifgpu) gpu_init(rank);
#endif

#ifdef SERIAL
   mem_check(ifgpu);
#else
   mem_check(ifgpu, world);
   if(schd.perfcomm) perfcomm<double>(world, 1ULL<<schd.perfcomm);
#endif

   if(schd.ctns.qkind == "rZ2"){
      RDM<ctns::qkind::qZ2,double>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      RDM<ctns::qkind::qZ2,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rN"){
      RDM<ctns::qkind::qN,double>(schd);
   }else if(schd.ctns.qkind == "cN"){
      RDM<ctns::qkind::qN,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      RDM<ctns::qkind::qNSz,double>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      RDM<ctns::qkind::qNSz,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      RDM<ctns::qkind::qNK,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNS"){
      RDM<ctns::qkind::qNS,double>(schd);
   }else if(schd.ctns.qkind == "cNS"){
      RDM<ctns::qkind::qNS,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for prop!");
   } // qkind

#ifdef GPU
   if(schd.ctns.alg_renorm>10){
      gpu_finalize();
   }
#endif

   // cleanup 
   if(rank == 0) tools::finish("RDM");	   
   return 0;
}
