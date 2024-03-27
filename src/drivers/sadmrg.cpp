#include <iostream>
#include <iomanip>
#include <string>
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
void SADMRG(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in SADMRG!");
   }

   // initialization: two options
   // 1. load from a given NSA-MPS
   // 2. load from a restart calculations
   ctns::comb<Qm,Tm> icomb;
   // convert from SCI or load from files
   if(rank == 0){
      // dealing with topology 
      icomb.topo.read(schd.ctns.topology_file);
      icomb.topo.print();
      if(schd.ctns.restart_sweep == 0){
         
         // initialize RCF 
         auto rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
         if(schd.ctns.tosu2){ 
            ctns::comb<ctns::qkind::qNSz,Tm> icomb_NSz;
            icomb_NSz.topo = icomb.topo;
            if(!schd.ctns.rcanon_load){
               // from SCI wavefunction
               onspace sci_space;
               vector<double> es;
               linalg::matrix<Tm> vs;
               auto ci_file = schd.scratch+"/"+schd.sci.ci_file;	   
               fci::ci_load(sci_space, es, vs, ci_file);
               // truncate CI coefficients
               fci::ci_truncate(sci_space, vs, schd.ctns.maxdets);
               ctns::rcanon_init(icomb_NSz, sci_space, vs, schd.ctns.rdm_svd,
                     schd.ctns.thresh_proj, schd.ctns.thresh_ortho);
               ctns::rcanon_save(icomb_NSz, rcanon_file);
            }else{
               ctns::rcanon_load(icomb_NSz, rcanon_file); // user defined rcanon_file
            } // rcanon_load

            // convert to SU2 symmetry via sweep projection
            ctns::rcanon_tosu2(icomb_NSz, icomb, schd.twos, schd.ctns.thresh_tosu2);
            ctns::rcanon_save(icomb, rcanon_file+"_su2");

         }else{
            ctns::rcanon_load(icomb, rcanon_file); // user defined rcanon_file
         }

      }else{
         // restart a broken calculation from disk
         auto rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.ctns.restart_sweep-1);
         if(schd.ctns.restart_sweep > schd.ctns.maxsweep){
            std::cout << "error: restart_sweep exceed maxsweep!" << std::endl;
            std::cout << " restart_sweep=" << schd.ctns.restart_sweep
                      << " maxsweep=" << schd.ctns.maxsweep
                      << std::endl;
            exit(1);
         }
         ctns::rcanon_load(icomb, rcanon_file);
      }
      
      ctns::rcanon_check(icomb, schd.ctns.thresh_ortho);
   } // rank 0

   if(schd.ctns.task_init) return; // only perform initialization (converting to CTNS)

#ifndef SERIAL
   if(size > 1){
      mpi_wrapper::broadcast(schd.world, icomb, 0);
      icomb.world = schd.world;
   }
#endif

   // compute sdiag
   if(schd.ctns.task_sdiag){
      // parallel sampling can be implemented in future (should be very simple)!
      if(rank == 0){
         int iroot  = schd.ctns.iroot;
         int nsample = schd.ctns.nsample;
         int ndetprt = schd.ctns.ndetprt; 
         double Sd = ctns::rcanon_Sdiag_sample(icomb, iroot, nsample, ndetprt);
      }
   }

   // compute hamiltonian or optimize ctns by dmrg algorithm
   if(schd.ctns.task_ham || schd.ctns.task_opt){
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
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      if(schd.ctns.task_ham && schd.ctns.restart_bond == 0){ // restart_bond require site_ibondN.info in existing scratch 
         io::remove_scratch(scratch, (rank == 0)); // start a new scratch
      }
      io::create_scratch(scratch, (rank == 0));
      // compute hamiltonian 
      if(schd.ctns.task_ham){
         auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch); 
         if(rank == 0){
            Hij.print("Hij",8);

            const bool debug_Hij = true;
            if(debug_Hij){
               ctns::comb<ctns::qkind::qNSz,Tm> icomb_NSz;
               ctns::rcanon_tononsu2(icomb, icomb_NSz);
               auto Hij_NSz = ctns::get_Hmat(icomb_NSz, int2e, int1e, ecore, schd, scratch);
               auto diff = (Hij - Hij_NSz).normF();
               std::cout << "diff=" << diff << std::endl; 
            }
            
            auto Sij = ctns::get_Smat(icomb);
            Sij.print("Sij");
         }
      }
      // optimization from current RCF
      if(schd.ctns.task_opt){
         ctns::sweep_opt(icomb, int2e, int1e, ecore, schd, scratch);
      }
   } // ham || opt
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
      if(rank == 0) std::cout << "\ncheck input again, there is no task for SADMRG!" << std::endl;
      return 0;
   }

   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

#ifdef GPU
   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
      gpu_init(rank);
   }
#endif

   if(schd.ctns.qkind == "rNS"){
      SADMRG<ctns::qkind::qNS,double>(schd);
   }else if(schd.ctns.qkind == "cNS"){
      SADMRG<ctns::qkind::qNS,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for sadmrg!");
   } // qkind

#ifdef GPU
   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
      gpu_finalize();
   }
#endif

   // cleanup 
   if(rank == 0){
      tools::finish("SADMRG");	   
   }else{
      // NOTE: scratch should be removed manually!
      //io::remove_scratch(schd.scratch, (rank == 0)); 
   }
   return 0;
}
