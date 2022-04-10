#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_twodot_renorm.h"
#include "sweep_twodot_hdiag.h"
#include "sweep_twodot_local.h"
#include "sweep_twodot_sigma.h"
#include "symbolic_formulae_twodot.h"
#include "symbolic_preprocess.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"

namespace ctns{

// twodot optimization algorithm
template <typename Km>
void sweep_twodot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
	          oper_stack<typename Km::dtype>& qops_stack,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore,
		  const std::string scratch){
   using Tm = typename Km::dtype;
   int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
   rank = icomb.world.rank();
   size = icomb.world.size();
#endif   
#ifdef _OPENMP
   maxthreads = omp_get_max_threads();
#endif
   const bool debug = (rank==0);
   if(debug){
      std::cout << "ctns::sweep_twodot"
	        << " alg_hvec=" << schd.ctns.alg_hvec
		<< " alg_renorm=" << schd.ctns.alg_renorm
                << " mpisize=" << size
	        << " maxthreads=" << maxthreads 
	        << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. check partition
   const auto& dbond = sweeps.seq[ibond];
   const bool ifNC = icomb.topo.check_partition(2, dbond, debug);

   // 1. load operators 
   auto fneed = icomb.topo.get_fqops(2, dbond, scratch, debug);
   qops_stack.fetch(fneed, debug);
   const auto& lqops  = qops_stack(fneed[0]);
   const auto& rqops  = qops_stack(fneed[1]);
   const auto& c1qops = qops_stack(fneed[2]);
   const auto& c2qops = qops_stack(fneed[3]);
   if(debug){
      std::cout << "qops info: rank=" << rank << std::endl;
      lqops.print("lqops");
      rqops.print("rqops");
      c1qops.print("c1qops");
      c2qops.print("c2qops");
      size_t tsize = lqops.size()+rqops.size()+c1qops.size()+c2qops.size();
      std::cout << " qops(tot)=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
   }
   timing.ta = tools::get_time();

   // 2. twodot wavefunction
   //	 \ /
   //   --*--
   const auto& ql = lqops.qket;
   const auto& qr = rqops.qket;
   const auto& qc1 = c1qops.qket;
   const auto& qc2 = c2qops.qket;
   auto sym_state = get_qsym_state(Km::isym, schd.nelec, schd.twoms);
   stensor4<Tm> wf(sym_state, ql, qr, qc1, qc2);
   if(debug) wf.print("wf"); 
 
   // 3. Davidson solver for wf
   int nsub = wf.size();
   int neig = sweeps.nroots;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);
   
   // 3.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   twodot_Hdiag(lqops, rqops, c1qops, c2qops, ecore, wf, diag, size, rank);
#ifndef SERIAL
   // reduction of partial Hdiag: no need to broadcast, if only rank=0 
   // executes the preconditioning in Davidson's algorithm
   if(size > 1){
      std::vector<double> diag2(nsub);
      boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
      diag = std::move(diag2);
   }
#endif 
   timing.tb = tools::get_time();

   // 3.2 Solve local problem: Hc=cE
   const oper_dictmap<Tm> qops_dict = {{"l",lqops},
	   		 	       {"r",rqops},
	   			       {"c1",c1qops},
				       {"c2",c2qops}};
   std::map<qsym,qinfo4<Tm>> info_dict;
   const bool preprocess = schd.ctns.alg_hvec==2 || schd.ctns.alg_hvec==3;
   size_t opsize, wfsize, tmpsize, worktot;
   opsize = preprocess_opsize(qops_dict);
   wfsize = preprocess_wfsize(wf.info, info_dict);
   if(schd.ctns.alg_hvec == 2){
      tmpsize = opsize + 3*wfsize;
   }else if(schd.ctns.alg_hvec == 3){
      tmpsize = opsize + 4*wfsize;
   }
   worktot = maxthreads*tmpsize;
   if(preprocess && debug){
      std::cout << "preprocess for Hx:"
                << " opsize=" << opsize 
                << " wfsize=" << wfsize 
                << " worktot=" << worktot 
                << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                << ":" << tools::sizeGB<Tm>(worktot) << "GB"
                << std::endl; 
   }
   std::string fname;
   if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
	      			     + "_"+std::to_string(isweep)
	                             + "_"+std::to_string(ibond)+".txt";
   HVec_type<Tm> HVec;
   Hx_functors<Tm> Hx_funs;
   symbolic_task<Tm> H_formulae;
   bipart_task<Tm> H_formulae2;
   Tm* workspace;
   using std::placeholders::_1;
   using std::placeholders::_2;
   if(schd.ctns.alg_hvec == 0){
      Hx_funs = twodot_Hx_functors(lqops, rqops, c1qops, c2qops, 
                                   int2e, ecore, wf, size, rank);
      HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 1){
      H_formulae = symbolic_formulae_twodot(lqops, rqops, c1qops, c2qops, 
		                            int2e, size, rank, fname,
			                    schd.ctns.sort_formulae);
      HVec = bind(&ctns::symbolic_Hx<Tm,stensor4<Tm>>, _1, _2, std::cref(H_formulae),
        	  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 2){ 
      H_formulae = symbolic_formulae_twodot(lqops, rqops, c1qops, c2qops, 
		                            int2e, size, rank, fname,
			                    schd.ctns.sort_formulae);
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx2<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
		  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		  std::ref(workspace));
   }else if(schd.ctns.alg_hvec == 3){
      H_formulae2 = symbolic_formulae_twodot2(lqops, rqops, c1qops, c2qops, 
		                              int2e, size, rank, fname,
			                      schd.ctns.sort_formulae);
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx3<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
		  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		  std::ref(workspace));
   } // alg_hvec
   oper_timer.clear();
   twodot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, dbond, wf);
   if(preprocess) delete[] workspace;
   if(debug){
      sweeps.print_eopt(isweep, ibond);
      if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
   }
   timing.tc = tools::get_time();

   // 3. decimation & renormalize operators
   auto frop = icomb.topo.get_frop(dbond, scratch, debug);
   twodot_renorm(schd, sweeps, isweep, ibond, icomb, vsol, wf, qops_stack(frop), 
		 lqops, rqops, c1qops, c2qops, int2e, int1e, scratch);
   timing.tf = tools::get_time();
  
   // 4. save on disk 
   qops_stack.save(frop, debug);

   timing.t1 = tools::get_time();
   if(debug){
      tools::timing("ctns::sweep_twodot", timing.t0, timing.t1);
      timing.analysis("time_local");
   }
}

} // ctns

#endif
