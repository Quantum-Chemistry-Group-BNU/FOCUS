#include <iostream>
#include <iomanip>
#include <string>
#include "../io/io.h"
#include "../io/input.h"
#include "../ci/ci_header.h"
#include "../ctns/ctns_header.h"
#include "../postmps/postmps_header.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GPU
#include "../gpu/gpu_env.h"
#endif

using namespace std;
using namespace fock;

template <typename Km>  
void postMPS(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   using Tm = typename Km::dtype;
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in postMPS!");
   }

   if(schd.postmps.task_ovlp){
      ctns::mps_ovlp<Km>(schd);
   }
   if(schd.postmps.task_cicoeff){
      ctns::mps_cicoeff<Km>(schd);
   }
   if(schd.postmps.task_sdiag){
      ctns::mps_sdiag<Km>(schd);
   }

   /*
   // compute hamiltonian or optimize ctns by dmrg algorithm
   if(schd.ctns.task_ham || schd.ctns.task_opt || schd.ctns.task_vmc){
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
   if(!schd.postmps.run){
      if(rank == 0) std::cout << "\ncheck input again, there is no task for POSTMPS!" << std::endl;
      return 0;
   }

   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

//#ifdef GPU
//   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
//      gpu_init(rank);
//   }
//#endif

   if(schd.postmps.qkind == "rZ2"){
      postMPS<ctns::qkind::rZ2>(schd);
   }else if(schd.postmps.qkind == "cZ2"){
      postMPS<ctns::qkind::cZ2>(schd);
   }else if(schd.postmps.qkind == "rN"){
      postMPS<ctns::qkind::rN>(schd);
   }else if(schd.postmps.qkind == "cN"){
      postMPS<ctns::qkind::cN>(schd);
   }else if(schd.postmps.qkind == "rNSz"){
      postMPS<ctns::qkind::rNSz>(schd);
   }else if(schd.postmps.qkind == "cNSz"){
      postMPS<ctns::qkind::cNSz>(schd);
   }else if(schd.postmps.qkind == "cNK"){
      postMPS<ctns::qkind::cNK>(schd);
   }else{
      tools::exit("error: no such qkind for postmps!");
   } // qkind

//#ifdef GPU
//   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
//      gpu_finalize();
//   }
//#endif

   // cleanup 
   if(rank == 0){
      tools::finish("postMPS");	   
   }else{
      // NOTE: scratch should be removed manually!
      //io::remove_scratch(schd.scratch, (rank == 0)); 
   }
   return 0;
}
