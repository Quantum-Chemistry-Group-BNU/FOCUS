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

namespace ctns{

// twodot optimization algorithm
template <typename Km>
void sweep_twodot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore,
		  const std::string scratch){
   int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
   rank = icomb.world.rank();
   size = icomb.world.size();
#endif   
#ifdef _OPENMP
   maxthreads = omp_get_max_threads();
#endif
   if(rank == 0){
      std::cout << "ctns::sweep_twodot"
	        << " alg_hvec=" << schd.ctns.alg_hvec
		<< " alg_renorm=" << schd.ctns.alg_renorm
                << " mpisize=" << size
	        << " maxthreads=" << maxthreads 
	        << std::endl;
   }
   const int isym = Km::isym;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. check partition
   const auto& dbond = sweeps.seq[ibond];
   const auto& p0 = dbond.p0;
   const auto& p1 = dbond.p1;
   const auto& p = dbond.p;
   const auto& cturn = dbond.cturn;
   std::vector<int> suppl, suppr, suppc1, suppc2;
   if(!cturn){
      //
      //       |    |
      //    ---p0---p1---
      //
      suppl  = icomb.topo.get_suppl(p0);
      suppr  = icomb.topo.get_suppr(p1);
      suppc1 = icomb.topo.get_suppc(p0);
      suppc2 = icomb.topo.get_suppc(p1);
   }else{
      //       |
      //    ---p1
      //       |
      //    ---p0---
      //
      suppl  = icomb.topo.get_suppl(p0);
      suppr  = icomb.topo.get_suppr(p0);
      suppc1 = icomb.topo.get_suppc(p1);
      suppc2 = icomb.topo.get_suppr(p1);
   }
   int sl = suppl.size();
   int sr = suppr.size();
   int sc1 = suppc1.size();
   int sc2 = suppc2.size();
   assert(sc1+sc2+sl+sr == icomb.topo.nphysical);
   if(rank == 0){
      std::cout << "support info: (sl,sr,sc1,sc2)=" 
		<< sl << "," << sr << "," << sc1 << "," << sc2
		<< std::endl;
      tools::print_vector(suppl , "suppl");
      tools::print_vector(suppr , "suppr");
      tools::print_vector(suppc1, "suppc1");
      tools::print_vector(suppc2, "suppc2");
   }

   // 1. load operators 
   using Tm = typename Km::dtype;
   oper_dict<Tm> lqops, rqops, c1qops, c2qops;
   if(!cturn){
      oper_load_qops(icomb, p0, scratch, "l", lqops , rank);
      oper_load_qops(icomb, p1, scratch, "r", rqops , rank);  
      oper_load_qops(icomb, p0, scratch, "c", c1qops, rank);
      oper_load_qops(icomb, p1, scratch, "c", c2qops, rank);
   }else{
      oper_load_qops(icomb, p0, scratch, "l", lqops , rank);
      oper_load_qops(icomb, p0, scratch, "r", rqops , rank);  
      oper_load_qops(icomb, p1, scratch, "c", c1qops, rank);
      oper_load_qops(icomb, p1, scratch, "r", c2qops, rank);
   }
   if(rank == 0){
      std::cout << "qops info: rank=" << rank << std::endl;
      lqops.print("lqops");
      rqops.print("rqops");
      c1qops.print("c1qops");
      c2qops.print("c2qops");
      size_t tsize = lqops.size()+rqops.size()+c1qops.size()+c2qops.size();
      std::cout << " optot=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
   }
   timing.ta = tools::get_time();

   // 2. twodot wavefunction
   //	 \ /
   //   --*--
   const auto& ql = lqops.qbra;
   const auto& qr = rqops.qbra;
   const auto& qc1 = c1qops.qbra;
   const auto& qc2 = c2qops.qbra;
   auto sym_state = get_qsym_state(isym, schd.nelec, schd.twoms);
   stensor4<Tm> wf(sym_state, ql, qr, qc1, qc2);
   if(rank == 0) wf.print("wf"); 
 
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
   HVec_type<Tm> HVec;
   Hx_functors<Tm> Hx_funs;
   symbolic_task<Tm> H_formulae;
   std::map<qsym,qinfo4<Tm>> info_dict;
   Tm* workspace;
   using std::placeholders::_1;
   using std::placeholders::_2;
   if(schd.ctns.alg_hvec == 0){
      Hx_funs = twodot_Hx_functors(lqops, rqops, c1qops, c2qops, 
                                   int2e, ecore, wf, size, rank);
      HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec > 0){
      std::string fname;
      if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
	      			        + "_"+std::to_string(isweep)
	                		+ "_"+std::to_string(ibond)+".txt"; 
      H_formulae = symbolic_formulae_twodot(lqops, rqops, c1qops, c2qops, 
		                            int2e, size, rank, fname);
      if(schd.ctns.alg_hvec == 1){
         HVec = bind(&ctns::symbolic_Hx<Tm,stensor4<Tm>>, _1, _2, std::cref(H_formulae),
           	     std::cref(qops_dict), std::cref(ecore),
                     std::ref(wf), std::cref(size), std::cref(rank));
      }else if(schd.ctns.alg_hvec == 2){
	 size_t opsize = preprocess_opsize(qops_dict);
	 size_t wfsize = preprocess_wf4size(wf.info, info_dict);
	 size_t tmpsize = opsize + 3*wfsize;
	 size_t worktot = maxthreads*tmpsize;
	 if(rank == 0){
	    std::cout << "preprocess:"
		      << " opsize=" << opsize << ":" << tools::sizeMB<Tm>(opsize) << "MB"
		      << " wfsize=" << wfsize << ":" << tools::sizeMB<Tm>(wfsize) << "MB"
		      << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
		      << std::endl; 
	 }
	 workspace = new Tm[worktot];
         HVec = bind(&ctns::symbolic_Hx2<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, std::cref(H_formulae),
           	     std::cref(qops_dict), std::cref(ecore), 
                     std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		     std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		     std::ref(workspace));
      }
   }
   oper_timer.clear();
   twodot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, dbond, wf);
   if(schd.ctns.alg_hvec == 2) delete[] workspace; 
   timing.tc = tools::get_time();
   if(rank == 0){ 
      sweeps.print_eopt(isweep, ibond);
      if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
   }

   // 3. decimation & renormalize operators
   twodot_renorm(schd, sweeps, isweep, ibond, icomb, vsol, wf, 
		 lqops, rqops, c1qops, c2qops, int2e, int1e, scratch);

   timing.t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::sweep_twodot", timing.t0, timing.t1);
      timing.analysis();
      sweeps.timing_global.accumulate(timing);
   }
}

} // ctns

#endif
