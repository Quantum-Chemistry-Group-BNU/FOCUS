#include <iostream>
#include <iomanip>
#include <string>
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
   size_t k = schd.sorb;
   size_t k2 = k*(k-1)/2;
   size_t k3 = k*(k-1)*(k-2)/6;
   if(rank == 0){
      const std::set<std::string> keys_avail = {"ova",
                                                "1p1h",
                                                "2p2h",
                                                "1p0h", "0p1h",
                                                "2p0h", "0p2h",
                                                "2p1h", "1p2h",
                                                "3p3h",
                                                "mrpt2"};
      std::cout << "\n" << tools::line_separator2 << std::endl;
      std::cout << "task_prop:";
      for(const auto& key : schd.ctns.task_prop){
         std::cout << " " << key;
         if(keys_avail.find(key) == keys_avail.end()){
            std::cout << "\nerror: keys_avail does not contain key=" << key << std::endl;
            exit(1);
         }
      }
      std::cout << std::endl;
      std::cout << " is_same=" << is_same << std::endl;
      std::cout << " MPS1:" << " nroot=" << icomb.get_nroots() 
         << " iroot=" << iroot << " file=" << schd.ctns.rcanon_file 
         << std::endl;
      assert(iroot <= icomb.get_nroots());
      std::cout << " MPS2:" << " nroot=" << icomb2.get_nroots() 
         << " jroot=" << jroot << " file=" << schd.ctns.rcanon2_file
         << std::endl;
      assert(jroot <= icomb2.get_nroots());
      // size of rdm
      std::cout << std::scientific << std::setprecision(2)
         << "size of rdms:" << std::endl;
      size_t rdm1size = k*k;
      std::cout << " size(rdm1)=" << rdm1size
         << ":" << tools::sizeMB<Tm>(rdm1size) << "MB"
         << ":" << tools::sizeGB<Tm>(rdm1size) << "GB" 
         << std::endl;
      size_t rdm2size = k2*k2;
      std::cout << " size(rdm2)=" << rdm2size
         << ":" << tools::sizeMB<Tm>(rdm2size) << "MB"
         << ":" << tools::sizeGB<Tm>(rdm2size) << "GB" 
         << std::endl;
      size_t rdm3size = k3*k3;
      std::cout << " size(rdm3)=" << rdm3size
         << ":" << tools::sizeMB<Tm>(rdm3size) << "MB"
         << ":" << tools::sizeGB<Tm>(rdm3size) << "GB" 
         << std::endl;
      std::cout << tools::line_separator2 << std::endl;
   } // rank-0

   // 0: overlap
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

   // also debug tdm using rdm_simple

   // 1: rdm1 
   if(tools::is_in_vector(schd.ctns.task_prop,"1p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      
      linalg::matrix<Tm> tdm1;
      if(schd.ctns.debug_rdm and rank == 0){
         // compared against CI
         tdm1.resize(k,k);
         tdm1.load_txt("rdm1ci."+std::to_string(iroot)+"."+std::to_string(jroot));
         std::cout << "trace=" << tdm1.trace() << std::endl;
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) boost::mpi::broadcast(icomb.world, tdm1, 0);
#endif
      }

      // compute rdm1 
      rdm1.resize(k,k);
      aux.rdm = tdm1; 
      ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1, aux);

      if(schd.ctns.debug_rdm and rank == 0){
         std::cout << "nrm2(tdm1)=" << tdm1.normF() << std::endl;
         std::cout << "nrm2(rdm1)=" << rdm1.normF() << std::endl;
         auto diff2 = tdm1 + rdm1;
         if(diff2.normF() < thresh) tdm1 *= -1; // may differ by sign
         auto diff1 = tdm1 - rdm1;
         std::cout << "diff[+]=" << diff2.normF() << " diff[-]=" << diff1.normF() << std::endl;
         if(diff1.normF() > thresh){
            for(int i=0; i<diff1.rows(); i++){
               for(int j=0; j<diff1.cols(); j++){
                  if(std::abs(diff1(i,j)) > thresh){
                     std::cout << "i,j=" << i << "," << j
                        << " tdm1=" << tdm1(i,j)
                        << " rdm1=" << rdm1(i,j)
                        << " diff=" << diff1(i,j)
                        << " spatial="
                        << i/2 << (i%2==0? "A+" : "B+")
                        << j/2 << (j%2==0? "A-" : "B-")
                        << std::endl;
                  }
               }
            }
         }
         if(Qm::ifabelian){
            auto rdm1b = ctns::rdm1_simple(icomb, icomb2, iroot, jroot);
            auto diff = rdm1 - rdm1b;
            std::cout << "diff|rdm1-rdm1b|=" << diff.normF() << std::endl; 
            assert(diff.normF() < thresh);
         }
      } // debug
   } // rdm1

   // 2: rdm2
   if(tools::is_in_vector(schd.ctns.task_prop,"2p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));

      linalg::matrix<Tm> tdm2;
      if(schd.ctns.debug_rdm and rank == 0){
         // compared against CI
         tdm2.resize(k2,k2);
         tdm2.load_txt("rdm2ci."+std::to_string(iroot)+"."+std::to_string(jroot));
         std::cout << "trace=" << tdm2.trace() << std::endl;
#ifndef SERIAL
         if(schd.ctns.debug_rdm and size > 1) boost::mpi::broadcast(icomb.world, tdm2, 0);
#endif
      }      
         
      // compute rdm2
      rdm2.resize(k2,k2);
      aux.rdm = tdm2;
      ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2, aux);

      if(schd.ctns.debug_rdm and rank == 0){
         std::cout << "nrm2(tdm2)=" << tdm2.normF() << std::endl;
         std::cout << "nrm2(rdm2)=" << rdm2.normF() << std::endl;
         auto diff2 = tdm2 + rdm2;
         if(diff2.normF() < thresh) tdm2 *= -1; // may differ by sign
         auto diff1 = tdm2 - rdm2;
         std::cout << "diff[+]=" << diff2.normF() << " diff[-]=" << diff1.normF() << std::endl;
         if(diff1.normF() > thresh){
            for(int i=0; i<diff1.rows(); i++){
               for(int j=0; j<diff1.cols(); j++){
                  if(std::abs(diff1(i,j)) > thresh){
                     auto p0p1 = tools::inverse_pair0(i);
                     auto q0q1 = tools::inverse_pair0(j);
                     auto p0 = p0p1.first;
                     auto p1 = p0p1.second;
                     auto q0 = q0q1.first;
                     auto q1 = q0q1.second;
                     std::cout << "i,j=" << i << "," << j
                        << " tdm2=" << tdm2(i,j)
                        << " rdm2=" << rdm2(i,j)
                        << " diff=" << diff1(i,j)
                        << " p0+p1+q1+q0=" << p0 << "," << p1 << ","
                        << q1 << "," << q0 
                        << " spatial="
                        << p0/2 << (p0%2==0? "A+" : "B+")
                        << p1/2 << (p1%2==0? "A+" : "B+")
                        << q1/2 << (q1%2==0? "A-" : "B-")
                        << q0/2 << (q0%2==0? "A-" : "B-")
                        << std::endl;
                  }
               }
            }
         }
         if(Qm::ifabelian){
            auto rdm2b = ctns::rdm2_simple(icomb, icomb2, iroot, jroot);
            auto diff = rdm2 - rdm2b;
            std::cout << "diff|rdm2-rdm2b|=" << diff.normF() << std::endl; 
            assert(diff.normF() < thresh);
         }
      } // debug

      // hamiltonian matrix elements if needed 
      if(rank == 0){
         integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         integral::load(int2e, int1e, ecore, schd.integral_file);
         assert(schd.sorb == int1e.sorb);
         rdm1 = get_rdm1_from_rdm2(rdm2, false, schd.nelec);
         auto Sij = rdm1.trace()/Tm(schd.nelec);
         auto Hij = get_etot(rdm2, rdm1, int2e, int1e) + Sij*ecore; 
         std::cout << "iroot=" << iroot << " jroot=" << jroot
            << " H(i,j)=" << std::fixed << std::setprecision(schd.ctns.outprec) << Hij 
            << std::endl;
         fock::get_cumulant(rdm2, rdm1);
      }
   } // rdm2

   // single-site entropy analysis: obtainable from 2-RDM
   if(rdm2.size()>0){
      auto s1 = fock::entropy1site(rdm2, rdm1);
      if(schd.ctns.debug_rdm and rank == 0){
         auto s1tmp = ctns::entropy1_simple(icomb, iroot);
         linalg::xaxpy(s1.size(), -1.0, s1.data(), s1tmp.data());
         double diff = linalg::xnrm2(s1tmp.size(), s1tmp.data());
         assert(diff < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"1p0h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(k,1);
      ctns::rdm_sweep("1p0h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm1tmp(k,k), rdm1tmp2(k,k), rdmtmp(k,1);
         // <psi|i+j|psi2> = <psi|i+(j|psi2>)
         ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1tmp, aux);
         auto image1 = icomb2.topo.get_image1();
         for(int j=0; j<k; j++){
            int kj = j/2, spin_j = j%2;
            auto icomb2_j = apply_opC(icomb2, kj, spin_j, 0); // j|psi2>
            /* 
            std::cout << std::endl; 
            std::cout << tools::line_separator2 << std::endl;
            std::cout << "LZD: j=" << j << " " << icomb.get_qsym_state() << " : "
               << icomb2_j.get_qsym_state()
               << std::endl;
            std::cout << tools::line_separator2 << std::endl;
            */
            ctns::rdm_sweep("1p0h", false, icomb, icomb2_j, schd, scratch, rdmtmp, aux);
            /*
            rdmtmp.print("rdmtmp_"+std::to_string(j));
            rdm1tmp.print("rdm1tmp");
            std::cout << tools::line_separator2 << std::endl;
            */
            int pj = 2*image1[kj] + spin_j; // map to the orbital index
            linalg::xcopy(k, rdmtmp.data(), rdm1tmp2.col(pj));
         }
         /*
         rdm1tmp.print("rdm1tmp");
         rdm1tmp2.print("rdm1tmp2");
         */
         auto rdm1diff = rdm1tmp2 - rdm1tmp;
         std::cout << "debug 1p0h via rdm1: |rdm1|=" << rdm1tmp.normF() << " |rdm1b|=" << rdm1tmp2.normF()
            << " |rdm1-rdm1b|=" << rdm1diff.normF() << std::endl;
         assert(rdm1diff.normF() < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"0p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(1,k);
      ctns::rdm_sweep("0p1h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm1tmp(k,k), rdm1tmp2(k,k), rdmtmp(1,k);
         // <psi|i+j|psi2> = (<psi|i+)j|psi2>
         ctns::rdm_sweep("1p1h", is_same, icomb, icomb2, schd, scratch, rdm1tmp, aux);
         auto image1 = icomb.topo.get_image1();
         for(int i=0; i<k; i++){
            int ki = i/2, spin_i = i%2;
            auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi>
            /* 
            std::cout << std::endl; 
            std::cout << tools::line_separator2 << std::endl;
            std::cout << "LZD: i=" << i << " " << icomb.get_qsym_state() << " : "
               << icomb_i.get_qsym_state()
               << std::endl;
            std::cout << tools::line_separator2 << std::endl;
            */
            ctns::rdm_sweep("0p1h", false, icomb_i, icomb2, schd, scratch, rdmtmp, aux);
            /*
            rdmtmp.print("rdmtmp_"+std::to_string(i));
            rdm1tmp.print("rdm1tmp");
            std::cout << tools::line_separator2 << std::endl;
            */
            int pi = 2*image1[ki] + spin_i; // map to the orbital index 
            linalg::xcopy(k, rdmtmp.data(), 1, rdm1tmp2.row(pi), k);
         }
         /*
         rdm1tmp.print("rdm1tmp");
         rdm1tmp2.print("rdm1tmp2");
         */
         auto rdm1diff = rdm1tmp2 - rdm1tmp;
         //rdm1diff.print("rdm1diff");
         std::cout << "debug 0p1h via rdm1: |rdm1|=" << rdm1tmp.normF() << " |rdm1b|=" << rdm1tmp2.normF()
            << " |rdm1-rdm1b|=" << rdm1diff.normF() << std::endl;
         assert(rdm1diff.normF() < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"2p0h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(k2,1);
      ctns::rdm_sweep("2p0h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2), rdmtmp(k2,1);
         // <psi|i+j+rs|psi2> (r<s) = <psi|i+j+(rs|psi2>)
         ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
         auto image1 = icomb2.topo.get_image1();
         for(int s=0; s<k; s++){
            for(int r=0; r<s; r++){
               int ks = s/2, spin_s = s%2;
               int kr = r/2, spin_r = r%2;
               auto icomb2_s = apply_opC(icomb2, ks, spin_s, 0); // s|psi2>
               auto icomb2_rs = apply_opC(icomb2_s, kr, spin_r, 0); // rs|psi2>
               ctns::rdm_sweep("2p0h", false, icomb, icomb2_rs, schd, scratch, rdmtmp, aux);
               int ps = 2*image1[ks] + spin_s; // map to the orbital index
               int pr = 2*image1[kr] + spin_r;
               auto psr = tools::canonical_pair0(ps,pr);
               Tm sgn = tools::sgn_pair0(ps,pr);  
               rdmtmp *= sgn;
               linalg::xcopy(k2, rdmtmp.data(), rdm2tmp2.col(psr));
            }
         }
         auto rdm2diff = rdm2tmp2 - rdm2tmp;
         std::cout << "debug 2p0h via rdm2: |rdm2|=" << rdm2tmp.normF() << " |rdm2b|=" << rdm2tmp2.normF()
            << " |rdm2-rdm2b|=" << rdm2diff.normF() << std::endl;
         assert(rdm2diff.normF() < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"0p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(1,k2);
      ctns::rdm_sweep("0p2h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2), rdmtmp(1,k2);
         // <psi|i+j+rs|psi2> (i>j,r<s) = (<psi|i+j+)rs|psi2>, |tmp>=ji|psi> (i>j)
         ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
         auto image1 = icomb.topo.get_image1();
         for(int i=0; i<k; i++){
            for(int j=0; j<i; j++){
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi2>
               auto icomb_ji = apply_opC(icomb_i, kj, spin_j, 0); // ji|psi2>
               ctns::rdm_sweep("0p2h", false, icomb_ji, icomb2, schd, scratch, rdmtmp, aux);
               int pi = 2*image1[ki] + spin_i; // map to the orbital index
               int pj = 2*image1[kj] + spin_j;
               auto pij = tools::canonical_pair0(pi,pj);
               Tm sgn = tools::sgn_pair0(pi,pj);
               rdmtmp *= sgn;
               linalg::xcopy(k2, rdmtmp.data(), 1, rdm2tmp2.row(pij), k2);
            }
         }
         auto rdm2diff = rdm2tmp2 - rdm2tmp;
         std::cout << "debug 0p2h via rdm2: |rdm2|=" << rdm2tmp.normF() << " |rdm2b|=" << rdm2tmp2.normF()
            << " |rdm2-rdm2b|=" << rdm2diff.normF() << std::endl;
         assert(rdm2diff.normF() < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"2p1h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(k2,k);
      ctns::rdm_sweep("2p1h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2), rdmtmp(k2,k);
         // <psi|i+j+rs|psi2> (r<s) = <psi|i+j+r(s|psi2>)
         ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
         auto image1 = icomb2.topo.get_image1();
         for(int s=0; s<k; s++){
            int ks = s/2, spin_s = s%2;
            auto icomb2_s = apply_opC(icomb2, ks, spin_s, 0); // s|psi2>
            ctns::rdm_sweep("2p1h", false, icomb, icomb2_s, schd, scratch, rdmtmp, aux);
            int ps = 2*image1[ks] + spin_s; // map to the orbital index
            for(int pr=0; pr<ps; pr++){
               auto psr = tools::canonical_pair0(ps,pr);
               linalg::xcopy(k2, rdmtmp.col(pr), rdm2tmp2.col(psr));
            }
         }
         auto rdm2diff = rdm2tmp2 - rdm2tmp;
         std::cout << "debug 2p1h via rdm2: |rdm2|=" << rdm2tmp.normF() << " |rdm2b|=" << rdm2tmp2.normF()
            << " |rdm2-rdm2b|=" << rdm2diff.normF() << std::endl;
         assert(rdm2diff.normF() < thresh);
      }
   }

   if(tools::is_in_vector(schd.ctns.task_prop,"1p2h")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      linalg::matrix<Tm> rdm(k,k2);
      ctns::rdm_sweep("1p2h", false, icomb, icomb2, schd, scratch, rdm, aux);
      // debug this case
      if(schd.ctns.debug_rdm){
         linalg::matrix<Tm> rdm2tmp(k2,k2), rdm2tmp2(k2,k2), rdmtmp(k,k2);
         // <psi|i+j+rs|psi2> (i>j,r<s) = (<psi|i+)j+rs|psi2>, |tmp>=i|psi> 
         ctns::rdm_sweep("2p2h", is_same, icomb, icomb2, schd, scratch, rdm2tmp, aux);
         auto image1 = icomb.topo.get_image1();
         for(int i=0; i<k; i++){
            int ki = i/2, spin_i = i%2;
            auto icomb_i = apply_opC(icomb, ki, spin_i, 0); // i|psi2>
            ctns::rdm_sweep("1p2h", false, icomb_i, icomb2, schd, scratch, rdmtmp, aux);
            int pi = 2*image1[ki] + spin_i; // map to the orbital index
            for(int pj=0; pj<pi; pj++){
               auto pij = tools::canonical_pair0(pi,pj);
               linalg::xcopy(k2, rdmtmp.row(pj), k, rdm2tmp2.row(pij), k2);
            }
         }
         auto rdm2diff = rdm2tmp2 - rdm2tmp;
         std::cout << "debug 1p2h via rdm2: |rdm2|=" << rdm2tmp.normF() << " |rdm2b|=" << rdm2tmp2.normF()
            << " |rdm2-rdm2b|=" << rdm2diff.normF() << std::endl;
         assert(rdm2diff.normF() < thresh);
      }
   }

   // 3: rdm3
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
         std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
      } 
      auto t1 = tools::get_time();
      std::cout << "total time for 3-RDM: " << tools::get_duration(t1-t0) << " S" << std::endl;
      // compared against other evaluation
      if(schd.ctns.debug_rdm and rank == 0 and Qm::ifabelian){
         auto rdm3b = ctns::rdm3_simple(icomb, icomb2, iroot, jroot);
         auto diff = rdm3 - rdm3b;
         std::cout << "debug 3p3h against rdm3_simple: |rdm3|=" << rdm3.normF() 
            << " |rdm3b|=" << rdm3b.normF()
            << " |rdm3-rdm3b|=" << diff.normF() << std::endl; 
         assert(diff.normF() < thresh);
      }
      // debug
      if(rank == 0){
         auto rdm2 = fock::get_rdm2_from_rdm3(rdm3);
         auto rdm1 = fock::get_rdm1_from_rdm2(rdm2);
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
         exit(1);
      }
   } // rdm3

   // save results
   if(rank == 0 and is_same){
      if(rdm1.size()>0 or rdm2.size()>0 or rdm3.size()>0){
         std::cout << "\nsave results for rdms:" << std::endl;
      }
      // save text
      if(rdm1.size()>0) rdm1.save_txt("rdm1mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm2.size()>0) rdm2.save_txt("rdm2mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm3.size()>0) rdm3.save_txt("rdm3mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      // compute natural orbitals
      if(rdm1.size()>0){
         auto natorbs = fock::get_natorbs(fock::get_rdm1s(rdm1));
         natorbs.save_txt("natorbs", schd.ctns.outprec);
      }
   }

   // 4: dsrg-mrpt2
   if(tools::is_in_vector(schd.ctns.task_prop,"mrpt2")){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      
      // load integrals
      const auto& nv2 = schd.ctns.nv2;
      const auto& nc2 = schd.ctns.nc2;
      aux.dsrg_load(nv2, nc2, k);

      auto image1 = icomb.topo.get_image1();
     
      double ept2v = 0.0, ept2c = 0.0, ept2 = 0.0;
      int alg_mrpt2 = 0;
      auto t0 = tools::get_time();
      if(alg_mrpt2 == 0){

         // compute rdm3 = <p+q+r+stu> (p>q>r,s<t<u)
         rdm3.resize(k3,k3);
         for(int i=0; i<k; i++){
            auto tx = tools::get_time();
            int ki = i/2, spin_i = i%2;
            auto icomb2_i = apply_opC(icomb2, ki, spin_i, 0); // i|psi> (u=i)
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
            std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
         } // i 
         auto tz = tools::get_time();
         std::cout << "total time for 3-RDM: " << tools::get_duration(tz-t0) << " S" << std::endl;

         // assemble e2 by contracting <x+y+z+wvu>
         for(int px=0; px<k; px++){
            for(int py=0; py<px; py++){
               for(int pz=0; pz<py; pz++){
                  for(int pu=0; pu<k; pu++){
                    for(int pv=0; pv<pu; pv++){
                       for(int pw=0; pw<pv; pw++){
                           auto pxyz = tools::canonical_triple0(px,py,pz);
                           auto puvw = tools::canonical_triple0(pu,pv,pw);
                           const auto& val = rdm3(pxyz,puvw);
                           auto pxy = tools::canonical_pair0(px,py);
                           auto pxz = tools::canonical_pair0(px,pz);
                           auto pyz = tools::canonical_pair0(py,pz);
                           auto puv = tools::canonical_pair0(pu,pv);
                           auto puw = tools::canonical_pair0(pu,pw);
                           auto pvw = tools::canonical_pair0(pv,pw);
                           // -hv[ew,xy]*tv[uv,ez]*<0|x+y+z+vu|w> (x>y>z)
                           auto vterm0 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxy), &aux.dsrg_tv(pz*nv2,puv));
                           auto vterm1 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxz), &aux.dsrg_tv(py*nv2,puv));
                           auto vterm2 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pyz), &aux.dsrg_tv(px*nv2,puv));
                           auto vterm3 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pxy), &aux.dsrg_tv(pz*nv2,puw));
                           auto vterm4 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pxz), &aux.dsrg_tv(py*nv2,puw));
                           auto vterm5 = linalg::xdot(nv2, &aux.dsrg_hv(pv*nv2,pyz), &aux.dsrg_tv(px*nv2,puw));
                           auto vterm6 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pxy), &aux.dsrg_tv(pz*nv2,pvw));
                           auto vterm7 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pxz), &aux.dsrg_tv(py*nv2,pvw));
                           auto vterm8 = linalg::xdot(nv2, &aux.dsrg_hv(pu*nv2,pyz), &aux.dsrg_tv(px*nv2,pvw));
                           ept2v += std::real((-vterm0 + vterm1 - vterm2 
                                               +vterm3 - vterm4 + vterm5
                                               -vterm6 + vterm7 - vterm8)*val); 
                           
                           /*
                           std::cout << "lzd: " << px << "," << py << "," << pz << "," 
                              << pu << "," << pv << "," << pw << std::endl;
                           std::cout << " vterm0=" << std::setprecision(8) << vterm0 << std::endl;
                           std::cout << " vterm0=" << vterm1 << std::endl;
                           std::cout << " vterm0=" << vterm2 << std::endl;
                           std::cout << " vterm0=" << vterm3 << std::endl;
                           std::cout << " vterm0=" << vterm4 << std::endl;
                           std::cout << " vterm0=" << vterm5 << std::endl;
                           std::cout << " vterm0=" << vterm6 << std::endl;
                           std::cout << " vterm0=" << vterm7 << std::endl;
                           std::cout << " vterm0=" << vterm8 << std::endl;
                           std::cout << " pxyz,puvw=" << pxyz << "," << puvw << " rdm3=" << val 
                              << " rdm3b=" << rdm3b(pxyz,puvw) << std::endl;
                           std::cout << " ept2v=" << ept2v << std::endl;
                           */
                           
                           // tc[mw,xy]*hc[uv,mz]*<0|x+y+z+vu|w> (x>y>z)
                           auto cterm0 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxy), &aux.dsrg_hc(pz*nc2,puv));
                           auto cterm1 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxz), &aux.dsrg_hc(py*nc2,puv));
                           auto cterm2 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pyz), &aux.dsrg_hc(px*nc2,puv));
                           auto cterm3 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pxy), &aux.dsrg_hc(pz*nc2,puw));
                           auto cterm4 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pxz), &aux.dsrg_hc(py*nc2,puw));
                           auto cterm5 = linalg::xdot(nc2, &aux.dsrg_tc(pv*nc2,pyz), &aux.dsrg_hc(px*nc2,puw));
                           auto cterm6 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pxy), &aux.dsrg_hc(pz*nc2,pvw));
                           auto cterm7 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pxz), &aux.dsrg_hc(py*nc2,pvw));
                           auto cterm8 = linalg::xdot(nc2, &aux.dsrg_tc(pu*nc2,pyz), &aux.dsrg_hc(px*nc2,pvw));
                           ept2c += std::real((cterm0 - cterm1 + cterm2
                                              -cterm3 + cterm4 - cterm5
                                              +cterm6 - cterm7 + cterm8)*val);
                        } // w
                     } // v
                  } // u
               } // z
            } // y
         } // x 

      }else if(alg_mrpt2 == 1){

         for(int i=0; i<k; i++){
            auto tx = tools::get_time();
            int ki = i/2, spin_i = i%2;
            auto icomb2_i = apply_opC(icomb2, ki, spin_i, 0); // i|psi> (u=i)
            linalg::matrix<Tm> rdm32(k3,k2);
            ctns::rdm_sweep("3p2h", false, icomb, icomb2_i, schd, scratch, rdm32, aux);
            // copy data to rdm3 <Psi_0|p+q+r+st|Psi_i>
            int pw = 2*image1[ki] + spin_i; // map to the orbital index
            for(int px=0; px<k; px++){
              for(int py=0; py<px; py++){
                for(int pz=0; pz<py; pz++){
                   for(int pu=0; pu<k; pu++){
                      for(int pv=0; pv<pu; pv++){
                         auto pxyz = tools::canonical_triple0(px,py,pz);
                         auto puv = tools::canonical_pair0(pu,pv);
                         const auto& val = rdm32(pxyz,puv);
                         auto pxy = tools::canonical_pair0(px,py);
                         auto pxz = tools::canonical_pair0(px,pz);
                         auto pyz = tools::canonical_pair0(py,pz);
                         // -hv[ew,xy]*tv[uv,ez]*<0|x+y+z+vu|w> (x>y>z)
                         auto vterm0 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxy), &aux.dsrg_tv(pz*nv2,puv));
                         auto vterm1 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pxz), &aux.dsrg_tv(py*nv2,puv));
                         auto vterm2 = linalg::xdot(nv2, &aux.dsrg_hv(pw*nv2,pyz), &aux.dsrg_tv(px*nv2,puv));
                         ept2v += std::real((-vterm0 + vterm1 - vterm2)*val); 
                         // tc[mw,xy]*hc[uv,mz]*<0|x+y+z+vu|w> (x>y>z)
                         auto cterm0 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxy), &aux.dsrg_hc(pz*nc2,puv));
                         auto cterm1 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pxz), &aux.dsrg_hc(py*nc2,puv));
                         auto cterm2 = linalg::xdot(nc2, &aux.dsrg_tc(pw*nc2,pyz), &aux.dsrg_hc(px*nc2,puv));
                         ept2c += std::real((cterm0 - cterm1 + cterm2)*val);
                      } // v
                   } // u
                } // z
              } // y
            } // x
            auto ty = tools::get_time();
            std::cout << " i=" << i << " time=" << tools::get_duration(ty-tx) << " S" << std::endl;
         } // i

      } // alg_mrpt2
      auto t1 = tools::get_time();

      ept2 = ept2v + ept2c;
      std::cout << "\nept2v = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2v << std::endl; 
      std::cout << "ept2c = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2c << std::endl; 
      std::cout << "ept2  = " << std::fixed << std::setprecision(schd.ctns.outprec) << ept2 << std::endl; 
      std::cout << "total time for PT2: " << tools::get_duration(t1-t0) << " S" << std::endl;
   } // pt2
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

#ifdef GPU
   if(schd.ctns.alg_renorm>10){
      gpu_init(rank);
   }
#endif

#ifndef SERIAL
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
