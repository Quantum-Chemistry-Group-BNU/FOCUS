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
void MPSPROP(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in MPSPROP!");
   }

   //---------------------------------------------------------------------
   // The driver mpsprop only support MPS with the same kind (topo,Qm,Tm)
   //---------------------------------------------------------------------

   // initialization of MPS
   bool exist1 = false;
   ctns::comb<Qm,Tm> icomb;
   if(schd.ctns.rcanon_file.size()>0){
      ctns::comb_load(icomb, schd, schd.ctns.rcanon_file);
      assert(icomb.topo.ifmps);
      if(rank == 0){
         icomb.display_shape();
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
         if(schd.ctns.savebin) ctns::rcanon_savebin(icomb2, schd.scratch+"/"+schd.ctns.rcanon2_file);
      }
   }else{
      auto jcomb = icomb;
      icomb2 = std::move(jcomb);
      is_same = true;
   }

   // display task_prop
   if(rank == 0){
      std::map<int,std::string> tasks = {{0,"overlap"},{1,"rdm1"},{2,"rdm2"}};
      std::cout << "\n" << tools::line_separator2 << std::endl;
      std::cout << "task_prop:";
      for(const auto& key : schd.ctns.task_prop){
         std::cout << " " << tasks.at(key);
      }
      std::cout << std::endl;
      std::cout << " MPS1 = " << schd.ctns.rcanon_file 
         << " nroot=" << icomb.get_nroots() 
         << " iroot=" << schd.ctns.iroot 
         << std::endl;
      std::cout << " MPS2 = " << schd.ctns.rcanon2_file 
         << " nroot=" << icomb2.get_nroots() 
         << " jroot=" << schd.ctns.jroot 
         << std::endl;
      std::cout << tools::line_separator2 << std::endl;
      assert(schd.ctns.iroot <= icomb.get_nroots());
      assert(schd.ctns.jroot <= icomb2.get_nroots());
   }

   // 0: overlap
   if(tools::is_in_vector(schd.ctns.task_prop,0)){
      if(rank == 0){
         auto Sij = get_Smat(icomb, icomb2);
         std::cout << std::endl;
         Sij.print("<MPS1|MPS2>", schd.ctns.outprec);
      }
   } 
 
   // 1: rdm1 
   if(tools::is_in_vector(schd.ctns.task_prop,1)){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      // compute rdm1 
      int k = 2*icomb.get_nphysical();
      linalg::matrix<Tm> rdm1(k,k);
      ctns::get_rdm(1, is_same, icomb, icomb2, schd, scratch, rdm1); 
   }

   // 2: rdm2
   if(tools::is_in_vector(schd.ctns.task_prop,2)){
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      // compute rdm1 
      int k = 2*icomb.get_nphysical();
      int k2 = k*(k-1)/2;
      linalg::matrix<Tm> rdm2(k2,k2);
      ctns::get_rdm(2, is_same, icomb, icomb2, schd, scratch, rdm2); 
   }

/*
   if(tools::is_in_vector(schd.ctns.task_prop,2)){
      // read integral
      integral::two_body<Tm> int2e;
      integral::one_body<Tm> int1e;
      double ecore;
      if(rank == 0) integral::load(int2e, int1e, ecore, schd.integral_file);
#ifndef SERIAL
      if(size > 1){
         boost::mpi::broadcast(schd.world, ecore, 0);
         boost::mpi::broadcast(schd.world, int1e, 0);
         mpi_wrapper::broadcast(schd.world, int2e, 0);
      }
#endif
      // compute Hij
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0));
      io::create_scratch(scratch, (rank == 0));
      auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch); 
      if(rank == 0){
         std::cout << std::endl;
         Hij.print("<MPS1|H|MPS2>", schd.ctns.outprec);
      }
   }
*/

   /*
   // task
   if(schd.ctns.task_prop == 1){
   // reduce density matrix
   if(rank == 0){
   auto rdm1 = ctns::rdm1_simple(icomb, icomb, schd.ctns.iroot, schd.ctns.iroot);
   auto rdm2 = ctns::rdm2_simple(icomb, icomb, schd.ctns.iroot, schd.ctns.iroot);
   Tm etot = fock::get_etot(rdm2,rdm1,int2e,int1e) + ecore;
   cout << "etot(rdm)=" << setprecision(12) << etot << endl;
   }
   }else if(schd.ctns.task_prop == 2){
   // transition density matrix
   if(rank == 0){
   auto tdm1 = ctns::rdm1_simple(icomb, icomb2, schd.ctns.iroot, schd.ctns.jroot);
   auto tdm2 = ctns::rdm2_simple(icomb, icomb2, schd.ctns.iroot, schd.ctns.jroot);
   Tm Hij = fock::get_etot(tdm2,tdm1,int2e,int1e);
   auto smat = get_Smat(icomb, icomb2);
   Hij += smat(schd.ctns.iroot,schd.ctns.jroot)*ecore; 
   cout << "<i|H|j>(rdm)=" << setprecision(12) << Hij << endl;
   }
   }else if(schd.ctns.task_prop == 3){
   // single-site entropy analysis
   if(rank == 0){
   auto s1 = ctns::entropy1_simple(icomb, schd.ctns.iroot);
   }
   } // task_prop
   */
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
      if(rank == 0) std::cout << "\ncheck input again, there is no task for MPSPROP!" << std::endl;
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
      MPSPROP<ctns::qkind::qZ2,double>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      MPSPROP<ctns::qkind::qZ2,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rN"){
      MPSPROP<ctns::qkind::qN,double>(schd);
   }else if(schd.ctns.qkind == "cN"){
      MPSPROP<ctns::qkind::qN,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      MPSPROP<ctns::qkind::qNSz,double>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      MPSPROP<ctns::qkind::qNSz,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      MPSPROP<ctns::qkind::qNK,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNS"){
      MPSPROP<ctns::qkind::qNS,double>(schd);
   }else if(schd.ctns.qkind == "cNS"){
      MPSPROP<ctns::qkind::qNS,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for prop!");
   } // qkind

#ifdef GPU
   if(schd.ctns.alg_renorm>10){
      gpu_finalize();
   }
#endif

   // cleanup 
   if(rank == 0) tools::finish("MPSPROP");	   
   return 0;
}
