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
void PROPERTY(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in PROPERTY!");
   }

   //-----------------------------------------------------------------
   // The driver prop only support MPS with the same kind (topo,Qm,Tm)
   //-----------------------------------------------------------------

   // initialization of MPS
   ctns::comb<Qm,Tm> icomb;
   if(schd.ctns.rcanon_file.size()>0){
      ctns::comb_load(icomb, schd, schd.ctns.rcanon_file);
      icomb.display_shape();
      if(schd.ctns.savebin) ctns::rcanon_savebin(icomb, schd.scratch+"/"+schd.ctns.rcanon_file);
   }
   ctns::comb<Qm,Tm> icomb2;
   if(schd.ctns.rcanon2_file.size()>0){
      ctns::comb_load(icomb2, schd, schd.ctns.rcanon2_file);
      icomb2.display_shape();
      if(schd.ctns.savebin) ctns::rcanon_savebin(icomb, schd.scratch+"/"+schd.ctns.rcanon2_file);
   }

/*
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
      if(rank == 0) std::cout << "\ncheck input again, there is no task for PROPERTY!" << std::endl;
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
      PROPERTY<ctns::qkind::qZ2,double>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      PROPERTY<ctns::qkind::qZ2,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rN"){
      PROPERTY<ctns::qkind::qN,double>(schd);
   }else if(schd.ctns.qkind == "cN"){
      PROPERTY<ctns::qkind::qN,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      PROPERTY<ctns::qkind::qNSz,double>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      PROPERTY<ctns::qkind::qNSz,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      PROPERTY<ctns::qkind::qNK,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNS"){
      PROPERTY<ctns::qkind::qNS,double>(schd);
   }else if(schd.ctns.qkind == "cNS"){
      PROPERTY<ctns::qkind::qNS,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for prop!");
   } // qkind

#ifdef GPU
   if(schd.ctns.alg_renorm>10){
      gpu_finalize();
   }
#endif

   // cleanup 
   if(rank == 0) tools::finish("PROPERTY");	   
   return 0;
}
