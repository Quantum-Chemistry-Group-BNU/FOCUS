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
void CTNS(const input::schedule& schd){
   int rank = 0, size = 1;
#ifndef SERIAL
   rank = schd.world.rank();
   size = schd.world.size();
#endif
   // consistency check for dtype
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in CTNS!");
   }

   // CTNS 
   ctns::comb<Qm,Tm> icomb;
   std::string rcanon_file;
   // convert from SCI or load from files
   if(rank == 0){
      // dealing with topology 
      icomb.topo.read(schd.ctns.topology_file);
      icomb.topo.print();
      if(schd.ctns.restart_sweep == 0){
         
         // initialize RCF 
         if(schd.ctns.fromnosym){

            ctns::rcanon_fromnosym(icomb);
            rcanon_file = schd.scratch+"/rmps_sym";
            ctns::rcanon_save(icomb, rcanon_file);

         }else{

            if(schd.ctns.rcanon_file.empty()){
               // from SCI wavefunction
               rcanon_file = schd.scratch+"/rcanon_ci";
               onspace sci_space;
               vector<double> es;
               linalg::matrix<Tm> vs;
               auto ci_file = schd.scratch+"/"+schd.sci.ci_file;	   
               fci::ci_load(sci_space, es, vs, ci_file);
               // consistency check
               if(sci_space[0].size() != 2*icomb.get_nphysical()){
                  std::cout << "error: state.size is inconsistent with 2*nphysical in topo:"
                     << " state.size=" << sci_space[0].size() << " nphysical=" << icomb.get_nphysical()
                     << std::endl;
                  exit(1);  
               } 
               // truncate CI coefficients
               fci::ci_truncate(sci_space, vs, schd.ctns.maxdets);
               ctns::rcanon_init(icomb, sci_space, vs, schd.ctns.rdm_svd,
                     schd.ctns.thresh_proj, schd.ctns.thresh_ortho);
               ctns::rcanon_save(icomb, rcanon_file);
               // debug        
               const bool debug = false;
               if(debug){ 
                  // <CI|CTNS>
                  auto Sij_mix = ctns::rcanon_CIovlp(icomb, sci_space, vs);
                  Sij_mix.print("Sij_mix");
                  // HIJ
                  integral::two_body<Tm> int2e;
                  integral::one_body<Tm> int1e;
                  double ecore;
                  integral::load(int2e, int1e, ecore, schd.integral_file);
                  io::create_scratch(schd.scratch);
                  auto Hij_ci = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
                  Hij_ci.print("Hij_ci",8);
                  auto Hij_ctns = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, schd.scratch);
                  Hij_ctns.print("Hij_ctns",8);
                  double diffH = (Hij_ctns - Hij_ci).normF();
                  cout << "\ncheck diffH=" << diffH << endl;
                  const double thresh = 1.e-8;
                  if(diffH > thresh) tools::exit(string("error: diffH > thresh=")+to_string(thresh));
                  io::remove_scratch(schd.scratch);
                  exit(1);
               }
            }else{
               rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
               ctns::rcanon_load(icomb, rcanon_file); // user defined rcanon_file
            }

         } // nosym

      }else{
         // restart a broken calculation from disk
         rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.ctns.restart_sweep-1);
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

   // debug
   if(schd.ctns.task_expand){
      if(rank == 0){
         ctns::rcanon_Sdiag_exact(icomb, schd.ctns.iroot, schd.ctns.pthrd);
      }
   }

   // compute sdiag
   if(schd.ctns.task_sdiag){
      // TODOs: parallel sampling can be implemented in future (should be very simple)!
      if(rank == 0){
         ctns::rcanon_Sdiag_sample(icomb, schd.ctns.iroot, schd.ctns.nsample, schd.ctns.pthrd);
      }
   }

   // compute hamiltonian or optimize ctns by dmrg algorithm
   if(schd.ctns.task_ham || schd.ctns.task_opt || schd.ctns.task_orbopt || schd.ctns.task_vmc){
      // read integral
      integral::two_body<Tm> int2e;
      integral::one_body<Tm> int1e;
      double ecore;
      if(rank == 0){
         integral::load(int2e, int1e, ecore, schd.integral_file);
         // consistency check
         if(int1e.sorb != icomb.get_nphysical()*2){
            std::cout << "error: int1e.sorb is inconsistent with 2*nphysical in topo:"
             << " sorb=" << int1e.sorb << " nphysical=" << icomb.get_nphysical()
             << std::endl;
            exit(1);  
         }
      }
#ifndef SERIAL
      if(size > 1){
         boost::mpi::broadcast(schd.world, ecore, 0);
         boost::mpi::broadcast(schd.world, int1e, 0);
         mpi_wrapper::broadcast(schd.world, int2e, 0);
      }
#endif
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      // restart_bond require site_ibondN.info in existing scratch 
      if((schd.ctns.task_ham || schd.ctns.task_opt) && schd.ctns.restart_bond == 0){ 
         io::remove_scratch(scratch, (rank == 0)); // start a new scratch
      }
      io::create_scratch(scratch, (rank == 0));
      // compute hamiltonian 
      if(schd.ctns.task_ham){
         auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch); 
         if(rank == 0){
            Hij.print("Hij",schd.ctns.outprec);
            auto Sij = ctns::get_Smat(icomb);
            Sij.print("Sij",schd.ctns.outprec);
         }
      }
      // optimization from current RCF
      if(schd.ctns.task_opt and schd.ctns.maxsweep>0){
         ctns::sweep_opt(icomb, int2e, int1e, ecore, schd, scratch);
      }
      // orbital optimization
      if(schd.ctns.task_orbopt){
         ctns::oodmrg(icomb, int2e, int1e, ecore, schd, scratch);
      }
      // vmc for estimation uncertainty
      if(schd.ctns.task_vmc){
         ctns::vmc_estimate(icomb, int2e, int1e, ecore, schd, scratch);
      }
   } // ham || opt || vmc
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
      if(rank == 0) std::cout << "\ncheck input again, there is no task for CTNS!" << std::endl;
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

#ifndef SERIAL
   if(schd.perfcomm) perfcomm<double>(world, 1ULL<<schd.perfcomm);
#endif

   if(schd.ctns.qkind == "rZ2"){
      CTNS<ctns::qkind::qZ2,double>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      CTNS<ctns::qkind::qZ2,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rN"){
      CTNS<ctns::qkind::qN,double>(schd);
   }else if(schd.ctns.qkind == "cN"){
      CTNS<ctns::qkind::qN,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      CTNS<ctns::qkind::qNSz,double>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      CTNS<ctns::qkind::qNSz,std::complex<double>>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      CTNS<ctns::qkind::qNK,std::complex<double>>(schd);
   }else{
      tools::exit("error: no such qkind for ctns!");
   } // qkind

#ifdef GPU
   if(schd.ctns.alg_hvec>10 || schd.ctns.alg_renorm>10){
      gpu_finalize();
   }
#endif

   // cleanup 
   if(rank == 0) tools::finish("CTNS");	   
   return 0;
}
