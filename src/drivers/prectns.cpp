#include <iostream>
#include <iomanip>
#include <string>
#include "../io/io.h"
#include "../io/input.h"
#include "../ci/ci_header.h"
#include "../ctns/ctns_header.h"

using namespace std;
using namespace fock;

template <typename Km>  
void preCTNS(const input::schedule& schd){
   int rank = 0; 
#ifndef SERIAL
   rank = schd.world.rank();
#endif
   // consistency check for dtype
   using Tm = typename Km::dtype;
   
   // preCTNS 
   ctns::comb<Km> icomb;
   // convert from SCI or load from files
   if(rank == 0){
      // dealing with topology 
      icomb.topo.read(schd.ctns.topology_file);
      icomb.topo.print();
   }
   // only perform initialization
   if(schd.ctns.task_init) return;
#ifndef SERIAL
   mpi_wrapper::broadcast(schd.world, icomb, 0);
   icomb.world = schd.world;
#endif

   // analyze
   ctns::preprocess_distribution(icomb, schd);
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
   boost::mpi::broadcast(world, schd, 0);
   schd.world = world;
#endif
   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

   if(schd.ctns.qkind == "rZ2"){
      preCTNS<ctns::qkind::rZ2>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      preCTNS<ctns::qkind::cZ2>(schd);
   }else if(schd.ctns.qkind == "rN"){
      preCTNS<ctns::qkind::rN>(schd);
   }else if(schd.ctns.qkind == "cN"){
      preCTNS<ctns::qkind::cN>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      preCTNS<ctns::qkind::rNSz>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      preCTNS<ctns::qkind::cNSz>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      preCTNS<ctns::qkind::cNK>(schd);
   }else{
      tools::exit("error: no such qkind for ctns!");
   } // qkind

#ifndef SERIAL
   world.barrier();
#endif
   if(rank > 0) io::remove_scratch(schd.scratch, (rank == 0));
   return 0;
}
