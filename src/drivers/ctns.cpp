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
void CTNS(const input::schedule& schd){
   int rank = 0; 
#ifndef SERIAL
   rank = schd.world.rank();
#endif
   // consistency check for dtype
   using Tm = typename Km::dtype;
   if((schd.dtype == 1) != tools::is_complex<Tm>()){
      tools::exit("error: inconsistent dtype in CTNS!");
   }
   
   // CTNS 
   ctns::comb<Km> icomb;
   // convert from SCI or load from files
   if(rank == 0){
      // dealing with topology 
      icomb.topo.read(schd.ctns.topology_file);
      icomb.topo.print();
      // initialize RCF 
      auto rcanon_file = schd.scratch+"/"+schd.ctns.rcanon_file;
      if(!schd.ctns.rcanon_load){
         // from SCI wavefunction
         onspace sci_space;
         vector<double> es;
         vector<vector<Tm>> vs;
         auto ci_file = schd.scratch+"/"+schd.sci.ci_file;	   
         fci::ci_load(sci_space, es, vs, ci_file);
         // truncate CI coefficients
      	 fci::ci_truncate(sci_space, vs, schd.ctns.maxdets);
/*
         // debug         
	 integral::two_body<Tm> int2e;
         integral::one_body<Tm> int1e;
         double ecore;
         integral::load(int2e, int1e, ecore, schd.integral_file);
         auto Hij_ci = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
	 Hij_ci.print("Hij_ci",8);
*/ 
         ctns::rcanon_init(icomb, sci_space, vs, schd.ctns.rdm_vs_svd,
			   schd.ctns.thresh_proj, schd.ctns.thresh_ortho);
         ctns::rcanon_save(icomb, rcanon_file);
      }else{
         ctns::rcanon_load(icomb, rcanon_file);
      }
      ctns::rcanon_check(icomb, schd.ctns.thresh_ortho);
   }
   // only perform initialization
   if(schd.ctns.task_init) return;
#ifndef SERIAL
   boost::mpi::broadcast(schd.world, icomb, 0);
   icomb.world = schd.world;
#endif

   if(schd.ctns.task_sdiag){
      // parallel sampling can be implemented in future (very simple)!
      if(rank == 0){
         int iroot  = schd.ctns.iroot;
         int nsample = schd.ctns.nsample;
         int ndetprt = schd.ctns.ndetprt; 
         double Sd = rcanon_Sdiag_sample(icomb, iroot, nsample, ndetprt);
      }
   }
   
   if(schd.ctns.task_ham or schd.ctns.task_opt){
      
      // preprocess
      //ctns::preprocess_oper(icomb);
      
      // read integral
      integral::two_body<Tm> int2e;
      integral::one_body<Tm> int1e;
      double ecore;
      if(rank == 0) integral::load(int2e, int1e, ecore, schd.integral_file);
#ifndef SERIAL
      boost::mpi::broadcast(schd.world, int1e, 0);
      boost::mpi::broadcast(schd.world, int2e, 0);
      boost::mpi::broadcast(schd.world, ecore, 0);
#endif
      // create scratch
      auto scratch = schd.scratch+"/sweep";
      io::remove_scratch(scratch, (rank == 0)); // cleanup
      io::create_scratch(scratch, (rank == 0));
      // compute hamiltonian 
      if(schd.ctns.task_ham){
         auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd, scratch); 
         if(rank == 0){
            Hij.print("Hij",8);
            auto Sij = ctns::get_Smat(icomb);
            Sij.print("Sij");
         }
      }
      // optimization from current RCF
      if(schd.ctns.task_opt){
         ctns::sweep_opt(icomb, int2e, int1e, ecore, schd, scratch);
         if(rank == 0){
            auto rcanon_file = schd.scratch+"/rcanon_new.info"; 
            ctns::rcanon_save(icomb, rcanon_file);
         }
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
   boost::mpi::broadcast(world, schd, 0);
   schd.world = world;
#endif
   // setup scratch directory
   if(rank > 0) schd.scratch += "_"+to_string(rank);
   io::create_scratch(schd.scratch, (rank == 0));

   if(schd.ctns.qkind == "rZ2"){
      CTNS<ctns::qkind::rZ2>(schd);
   }else if(schd.ctns.qkind == "cZ2"){
      CTNS<ctns::qkind::cZ2>(schd);
   }else if(schd.ctns.qkind == "rN"){
      CTNS<ctns::qkind::rN>(schd);
   }else if(schd.ctns.qkind == "cN"){
      CTNS<ctns::qkind::cN>(schd);
   }else if(schd.ctns.qkind == "rNSz"){
      CTNS<ctns::qkind::rNSz>(schd);
   }else if(schd.ctns.qkind == "cNSz"){
      CTNS<ctns::qkind::cNSz>(schd);
   }else if(schd.ctns.qkind == "cNK"){
      CTNS<ctns::qkind::cNK>(schd);
   }else{
      tools::exit("error: no such qkind for ctns!");
   } // qkind

#ifndef SERIAL
   world.barrier();
#endif
   if(rank > 0) io::remove_scratch(schd.scratch, (rank == 0));
   return 0;
}
