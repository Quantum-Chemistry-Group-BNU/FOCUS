#include <iostream>
#include <iomanip>
#include <string>
#include "io/io.h"
#include "io/input.h"
#include "ci/ci_header.h"
#include "vmc/vmc_header.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace fock;

template <typename Tm>  
void VMC(const input::schedule& schd,
         vmc::BaseAnsatz& wavefun){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   
   // from SCI wavefunction
   onspace sci_space;
   vector<double> es;
   linalg::matrix<Tm> vs;
   auto ci_file = schd.scratch+"/"+schd.ci.ci_file;	   
   fci::ci_load(sci_space, es, vs, ci_file);

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
     
   auto Hij_ci = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
   Hij_ci.print("Hij");

   // load
   if(schd.vmc.wf_load){ 
      auto wf_file = schd.scratch+"/"+schd.vmc.wf_file; 
      wavefun.load(wf_file);
   }else{
      wavefun.init(int1e.sorb, schd.vmc.nhiden, schd.vmc.iscale);
   }
   
   // optimization
   if(schd.vmc.exactopt){
      vmc::opt_exact(wavefun, int2e, int1e, ecore, schd);
   }else{
      vmc::opt_sample(wavefun, int2e, int1e, ecore, schd, sci_space);
   }
   
   // save
   auto wf_file = schd.scratch+"/vmc_new.info";
   wavefun.save(wf_file);
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
   if(!schd.vmc.run){
      if(rank == 0) std::cout << "\ncheck input again, there is no task for VMC!" << std::endl;
      return 0;
   }

   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

   // define wavefunction
   if(schd.vmc.ansatz == "irbm"){
      vmc::irbm wavefun;
      if(schd.dtype == 0){
         VMC<double>(schd, wavefun);
      }else if(schd.dtype == 1){
         VMC<complex<double>>(schd, wavefun);
      }
   }else if(schd.vmc.ansatz == "trbm"){
      vmc::trbm wavefun;
      if(schd.dtype == 0){
         VMC<double>(schd, wavefun);
      }else if(schd.dtype == 1){
         VMC<complex<double>>(schd, wavefun);
      }
   }else if(schd.vmc.ansatz == "rrbm"){
      vmc::rrbm wavefun;
      if(schd.dtype == 0){
         VMC<double>(schd, wavefun);
      }else if(schd.dtype == 1){
         VMC<complex<double>>(schd, wavefun);
      }
   }else if(schd.vmc.ansatz == "irbmcos"){
      vmc::irbmcos wavefun;
      if(schd.dtype == 0){
         VMC<double>(schd, wavefun);
      }else if(schd.dtype == 1){
         VMC<complex<double>>(schd, wavefun);
      }
   }else{
      std::cout << "error: no such ansatz=" << schd.vmc.ansatz << std::endl;
      exit(1);
   }

   // cleanup 
   if(rank == 0){
      tools::finish("VMC");
   }else{
      io::remove_scratch(schd.scratch, (rank == 0));
   }
   return 0;
}
