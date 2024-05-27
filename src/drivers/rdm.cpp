#include <iostream>
#include <iomanip>
#include <string>
#include "core/perfcomm.h"
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
   // The driver rdm only support MPS with the same kind (topo,Qm,Tm)
   //-----------------------------------------------------------------

   // initialization of MPS
   ctns::comb<Qm,Tm> icomb;
   if(schd.ctns.rcanon_file.size()>0){
      ctns::comb_load(icomb, schd, schd.ctns.rcanon_file);
      icomb.display_shape();
   }
   ctns::comb<Qm,Tm> icomb2;
   if(schd.ctns.rcanon2_file.size()>0){
      ctns::comb_load(icomb2, schd, schd.ctns.rcanon2_file);
      icomb2.display_shape();
   }

   // rdm task
   if(schd.ctns.task_rdm == 1){
      ctns::rdm1_simple(icomb, icomb, schd.ctns.iroot, schd.ctns.iroot);
      ctns::rdm2_simple(icomb, icomb, schd.ctns.iroot, schd.ctns.iroot);
   }else if(schd.ctns.task_rdm == 2){
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
/*
 69    // make_rdm2 from sparseH
 75    double etot = fock::get_etot(rdm2,int2e,int1e,ecore);
 76    cout << "etot(rdm)=" << setprecision(12) << etot << endl;
 77    assert(std::abs(etot-es1[0]) < 1.e-8);
 78
*/
   } // task_rdm
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
//   }else if(schd.ctns.qkind == "rNS"){
//      RDM<ctns::qkind::qNS,double>(schd);
//   }else if(schd.ctns.qkind == "cNS"){
//      RDM<ctns::qkind::qNS,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for rdm!");
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
