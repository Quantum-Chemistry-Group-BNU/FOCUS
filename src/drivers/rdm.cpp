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

   // initialization of MPS
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
   if(rank == 0){
      std::map<int,std::string> tasks = {{0,"overlap"},{1,"rdm1"},{2,"rdm2"}};
      std::cout << "\n" << tools::line_separator2 << std::endl;
      std::cout << "task_prop:";
      for(const auto& key : schd.ctns.task_prop){
         std::cout << " " << tasks.at(key);
      }
      std::cout << "   is_same=" << is_same << std::endl;
      std::cout << " MPS1:"
         << " nroot=" << icomb.get_nroots() 
         << " iroot=" << iroot 
         << " file=" << schd.ctns.rcanon_file 
         << std::endl;
      std::cout << " MPS2:" 
         << " nroot=" << icomb2.get_nroots() 
         << " jroot=" << jroot 
         << " file=" << schd.ctns.rcanon2_file
         << std::endl;
      std::cout << tools::line_separator2 << std::endl;
      assert(iroot <= icomb.get_nroots());
      assert(jroot <= icomb2.get_nroots());
   }
   const double thresh = 1.e-6;

   // 0: overlap
   if(tools::is_in_vector(schd.ctns.task_prop,0)){
      if(rank == 0){
         auto Sij = get_Smat(icomb, icomb2);
         std::cout << std::endl;
         Sij.print("<MPS1|MPS2>", schd.ctns.outprec);
      }
   } 

   // --- rdms ---
   int k = 2*icomb.get_nphysical();
   int k2 = k*(k-1)/2;
   linalg::matrix<Tm> rdm1;
   linalg::matrix<Tm> rdm2;

   // 1: rdm1 
   if(tools::is_in_vector(schd.ctns.task_prop,1)){
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
      ctns::rdm_sweep(1, is_same, icomb, icomb2, schd, scratch, rdm1, tdm1);

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
   if(tools::is_in_vector(schd.ctns.task_prop,2)){
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
      ctns::rdm_sweep(2, is_same, icomb, icomb2, schd, scratch, rdm2, tdm2);

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
         rdm1 = get_rdm1_from_rdm2(rdm2, false, schd.nelec);
         auto Sij = rdm1.trace()/Tm(schd.nelec);
         auto Hij = get_etot(rdm2, rdm1, int2e, int1e) + Sij*ecore; 
         std::cout << "iroot=" << iroot << " jroot=" << jroot
            << " H(i,j)=" << std::fixed << std::setprecision(schd.ctns.outprec) << Hij 
            << std::endl;
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

   // save results
   if(rank == 0 and is_same){
      if(rdm1.size()>0 or rdm2.size()>0) std::cout << "\nsave results for rdms:" << std::endl;
      if(rdm1.size()>0) rdm1.save_txt("rdm1mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm2.size()>0) rdm2.save_txt("rdm2mps."+std::to_string(iroot)+"."+std::to_string(iroot), schd.ctns.outprec);
      if(rdm1.size()>0){
         auto natorbs = fock::get_natorbs(fock::get_rdm1s(rdm1));
         natorbs.save_txt("natorbs", schd.ctns.outprec);
      }
   }
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
