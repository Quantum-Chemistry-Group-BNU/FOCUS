#include <iostream>
#include <iomanip>
#include <string>
#include "core/mem_status.h"
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
   std::string rcanon_file;
   // convert from SCI or load from files
   if(rank == 0){
      // dealing with topology 
      if(!schd.ctns.topology_file.empty()){
         icomb.topo.read(schd.ctns.topology_file);
      }else{
         icomb.topo.gen1d(schd.sorb/2);
      }
      icomb.topo.print();
      if(schd.ctns.restart_sweep == 0){

         // initialize RCF 
         if(!schd.ctns.inputconf.empty()){

            // from a single configuration
            fock::csfstate csf(schd.ctns.inputconf);
            // consistency check
            if(csf.norb() != schd.sorb/2 or csf.nelec() != schd.nelec or csf.twos() != schd.twos){
               std::cout << "error: (k,ne,ts)=" << csf.norb() << "," << csf.nelec() << "," << csf.twos()
                 << " of csf=" << csf << " is inconsistent with input (k,ne,ts)=" 
                 << schd.sorb/2 << "," << schd.nelec << "," << schd.twos
                 << std::endl;
               exit(1);
            }
            icomb = ctns::csf2samps<Tm>(icomb.topo, csf);
            rcanon_file = schd.scratch+"/rcanon_conf_su2"; 
            ctns::rcanon_save(icomb, rcanon_file);
         
         }else if(!schd.ctns.loadconfs.empty()){

            // from a list of configurations
            ctns::rcanon_loadconfs(icomb, schd.ctns.loadconfs); 
            rcanon_file = schd.scratch+"/rcanon_confs_su2";
            ctns::rcanon_save(icomb, rcanon_file);

         }else{ 

            if(schd.ctns.tosu2){

               ctns::comb<ctns::qkind::qNSz,Tm> icomb_NSz;
               icomb_NSz.topo = icomb.topo;
               if(schd.ctns.rcanon_file.empty()){
                  // from SCI wavefunction
                  rcanon_file = schd.scratch+"/rcanon_ci";
                  onspace sci_space;
                  vector<double> es;
                  linalg::matrix<Tm> vs;
                  auto ci_file = schd.scratch+"/"+schd.ci.ci_file;	   
                  fci::ci_load(sci_space, es, vs, ci_file);
                  // consistency check
                  if(sci_space[0].size() != schd.sorb){
                     std::cout << "error: state.size is inconsistent with sorb:"
                        << " state.size=" << sci_space[0].size() << " schd.sorb=" << schd.sorb
                        << std::endl;
                     exit(1);  
                  }
                  // truncate CI coefficients
                  ctns::rcanon_init(icomb_NSz, sci_space, vs, schd);
                  ctns::rcanon_save(icomb_NSz, rcanon_file);
               }else{
                  rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
                  ctns::rcanon_load(icomb_NSz, rcanon_file); // user defined rcanon_file
               } // rcanon_load

               // convert to SU2 symmetry via sweep projection
               ctns::rcanon_tosu2(icomb_NSz, icomb, schd.twos, schd.ctns.thresh_tosu2);
               rcanon_file += "_su2"; 
               ctns::rcanon_save(icomb, rcanon_file);

               // debug by checking the overlap
               const bool debug_convert_and_hmat = false;
               if(debug_convert_and_hmat){
                  ctns::comb<ctns::qkind::qNSz,Tm> icomb_NSz2;
                  auto sym_state = icomb_NSz.get_qsym_state(); 
                  ctns::rcanon_tononsu2(icomb, icomb_NSz2, sym_state.tm());
                  // HIJ
                  integral::two_body<Tm> int2e;
                  integral::one_body<Tm> int1e;
                  double ecore;
                  integral::load(int2e, int1e, ecore, schd.integral_file);
                  io::create_scratch(schd.scratch);
                  // compare 
                  auto Hij1 = ctns::get_Hmat(icomb_NSz, int2e, int1e, ecore, schd, schd.scratch);
                  auto Hij2 = ctns::get_Hmat(icomb_NSz2, int2e, int1e, ecore, schd, schd.scratch);
                  auto Hij3 = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, schd.scratch);
                  std::cout << "\ncompare:" << std::endl;
                  Hij1.print("icomb_NSz",10);
                  Hij2.print("icomb_NSz2",10);
                  Hij3.print("icomb",10);
                  exit(1);
               }

            }else{
               // load from a given file
               if(schd.ctns.rcanon_file.empty()){
                  std::cout << "error: tosu2=false and rcanon_file is empty!" << std::endl;
                  exit(1);
               }
               rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
               ctns::rcanon_load(icomb, rcanon_file); // user defined rcanon_file
            } // tosu2

         } // inputconf

      }else{
         // restart a broken calculation from disk
         rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.ctns.restart_sweep-1)+"_su2";
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
      icomb.display_size();
      // consistency check
      if(schd.sorb != 2*icomb.get_nphysical()){
         tools::exit("error in sadmrg: schd.sorb != 2*icomb.get_nphysical()");
      }
      if(schd.nelec != icomb.get_qsym_state().ne()){
         tools::exit("error in sadmrg: schd.nelec != icomb.get_qsym_state().ne()");
      }
      if(schd.twos != icomb.get_qsym_state().ts()){
         tools::exit("error in sadmrg: schd.twos != icomb.get_qsym_state().ts()");
      }
   } // rank 0

   if(schd.ctns.task_init) return; // only perform initialization (converting to CTNS)

#ifndef SERIAL
   if(size > 1){
      mpi_wrapper::broadcast(schd.world, icomb, 0);
      icomb.world = schd.world;
   }
#endif

   // debug
   if(schd.ctns.task_expand){
      if(rank == 0){
         if(!schd.ctns.detbasis){
            ctns::rcanon_Sdiag_exact(icomb, schd.ctns.iroot, "csf", schd.ctns.pthrd);
         }else{
            ctns::rcanon_Sdiag_exact(icomb, schd.ctns.iroot, "det", schd.ctns.pthrd); 
         }
      }
   }

   // compute sdiag
   if(schd.ctns.task_sdiag){
      // parallel sampling can be implemented in future (should be very simple)!
      if(rank == 0){
         if(!schd.ctns.detbasis){
            ctns::rcanon_Sdiag_sample(icomb, schd.ctns.iroot, schd.ctns.nsample, 
                  schd.ctns.pthrd, schd.ctns.nprt, schd.ctns.saveconfs);
            ctns::rcanon_listcoeff(icomb, schd.ctns.iroot, schd.ctns.thresh_cabs,
                  schd.ctns.saveconfs);
         }else{
            ctns::rcanon_Sdiag_sample_det(icomb, schd.ctns.iroot, schd.ctns.nsample, 
                  schd.ctns.pthrd, schd.ctns.nprt, schd.ctns.saveconfs);
            ctns::rcanon_listcoeff_det(icomb, schd.ctns.iroot, schd.ctns.thresh_cabs,
                  schd.ctns.saveconfs);
         }
      }
   }

   // convert to nonsu2
   if(schd.ctns.task_tononsu2){
      if(rank == 0){
         ctns::comb<ctns::qkind::qNSz,Tm> icomb_NSz;
         ctns::rcanon_tononsu2(icomb, icomb_NSz, schd.twom);
         rcanon_file += "_nonsu2"; 
         ctns::rcanon_save(icomb_NSz, rcanon_file);
         icomb.display_shape();
         icomb_NSz.display_shape();
         ctns::rcanon_Sdiag_sample(icomb_NSz, schd.ctns.iroot, schd.ctns.nsample, 
               schd.ctns.pthrd, schd.ctns.nprt, schd.ctns.saveconfs);
      }
   }

   // compute schmidt decomposition
   if(schd.ctns.task_schmidt){
      if(rank == 0){
         ctns::rcanon_schmidt(icomb, schd.ctns.iroot, schd.scratch+"/"+schd.ctns.schmidt_file);
      }
   }

   // compute hamiltonian or optimize ctns by dmrg algorithm
   if(schd.ctns.task_ham || schd.ctns.task_opt || schd.ctns.task_oodmrg){
      // read integral
      integral::two_body<Tm> int2e;
      integral::one_body<Tm> int1e;
      double ecore;
      if(rank == 0){
         integral::load(int2e, int1e, ecore, schd.integral_file);
         if(schd.sorb != int1e.sorb){
            tools::exit("error in sadmrg: schd.sorb != int1e.sorb");
         }
      }
#ifndef SERIAL
      if(size > 1){
         boost::mpi::broadcast(schd.world, ecore, 0);
         mpi_wrapper::broadcast(schd.world, int1e, 0);
         mpi_wrapper::broadcast(schd.world, int2e, 0);
      }
#endif
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      // restart_bond require data in existing scratch 
      if((schd.ctns.task_ham || schd.ctns.task_opt) && schd.ctns.restart_bond == -1){ 
         io::remove_scratch(scratch, (rank == 0)); // start a new scratch
      }
      io::create_scratch(scratch, (rank == 0));
      // compute hamiltonian 
      if(schd.ctns.task_ham and !schd.ctns.localrestart){
         auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch); 
         if(rank == 0){
            Hij.print("Hij",schd.ctns.outprec);
            auto Sij = ctns::get_Smat(icomb);
            Sij.print("Sij",schd.ctns.outprec);
         }
      }
      // optimization from current RCF
      if(schd.ctns.task_opt){
         ctns::sweep_opt(icomb, int2e, int1e, ecore, schd, scratch, schd.ctns.rcfprefix);
      }
      // orbital optimization
      if(schd.ctns.task_oodmrg){
         ctns::oodmrg(icomb, int2e, int1e, ecore, schd, scratch);
      }
   } // ham || opt || oodmrg
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

   const bool ifgpu = schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10;
#ifdef GPU
   if(ifgpu) gpu_init(rank);
#endif

#ifdef SERIAL
   mem_check(ifgpu);
#else
   mem_check(ifgpu, world);
   if(schd.perfcomm) perfcomm<double>(world, 1ULL<<schd.perfcomm);
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
   if(rank == 0) tools::finish("SADMRG");	   
   return 0;
}
