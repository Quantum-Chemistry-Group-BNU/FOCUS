#include <iostream>
#include <iomanip>
#include <string>
#include "../io/io.h"
#include "../io/input.h"
#include "../ci/ci_header.h"
#include "../ctns/ctns_header.h"
#include "../post/post_header.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GPU
#include "../gpu/gpu_env.h"
#endif

using namespace std;
using namespace fock;

template <typename Km>  
void POST(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   using Tm = typename Km::dtype;
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in POST!");
   }

   if(schd.post.task_ovlp){
      ctns::mps_ovlp<Km>(schd);
   }
   if(schd.post.task_cicoeff){
      ctns::mps_cicoeff<Km>(schd);
   }
   if(schd.post.task_sdiag){
      ctns::mps_sdiag<Km>(schd);
   }
   if(schd.post.task_expect){
      ctns::mps_expect<Km>(schd);
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
   if(!schd.post.run){
      if(rank == 0) std::cout << "\ncheck input again, there is no task for POST!" << std::endl;
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

   if(schd.post.qkind == "rZ2"){
      POST<ctns::qkind::rZ2>(schd);
   }else if(schd.post.qkind == "cZ2"){
      POST<ctns::qkind::cZ2>(schd);
   }else if(schd.post.qkind == "rN"){
      POST<ctns::qkind::rN>(schd);
   }else if(schd.post.qkind == "cN"){
      POST<ctns::qkind::cN>(schd);
   }else if(schd.post.qkind == "rNSz"){
      POST<ctns::qkind::rNSz>(schd);
   }else if(schd.post.qkind == "cNSz"){
      POST<ctns::qkind::cNSz>(schd);
   }else if(schd.post.qkind == "cNK"){
      POST<ctns::qkind::cNK>(schd);
   }else{
      tools::exit("error: no such qkind for POST!");
   } // qkind

//#ifdef GPU
//   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
//      gpu_finalize();
//   }
//#endif

   // cleanup 
   if(rank == 0){
      tools::finish("POST");	   
   }else{
      // NOTE: scratch should be removed manually!
      //io::remove_scratch(schd.scratch, (rank == 0)); 
   }
   return 0;
}
