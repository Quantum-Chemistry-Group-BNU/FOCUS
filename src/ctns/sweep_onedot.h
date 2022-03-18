#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_onedot_renorm.h"
#include "sweep_onedot_hdiag.h"
#include "sweep_onedot_local.h"
#include "sweep_onedot_sigma.h"
#include "symbolic_onedot_formulae.h"
#include "symbolic_sigma.h"
#include "symbolic_preprocess.h"
#include "symbolic_sigma2.h"

namespace ctns{

// onedot optimization algorithm
template <typename Km>
void sweep_onedot(const input::schedule& schd,
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
      std::cout << "ctns::sweep_onedot"
	        << " alg_hvec=" << schd.ctns.alg_hvec
		<< " alg_renorm=" << schd.ctns.alg_renorm
                << " mpisize=" << size
	        << " maxthreads=" << maxthreads 
	        << std::endl;
   }
   const int isym = Km::isym;
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // check partition 
   const auto& dbond = sweeps.seq[ibond];
   const auto& p = dbond.p;
   std::vector<int> suppl, suppr, suppc;
   suppl = icomb.topo.get_suppl(p);
   suppr = icomb.topo.get_suppr(p);
   suppc = icomb.topo.get_suppc(p); 
   int sl = suppl.size();
   int sr = suppr.size();
   int sc = suppc.size();
   assert(sc+sl+sr == icomb.topo.nphysical);
   if(rank == 0){ 
      std::cout << "support info: (sl,sr,sc)=" 
                << sl << "," << sr << "," << sc 
                << std::endl;
      tools::print_vector(suppl, "suppl");
      tools::print_vector(suppr, "suppr");
      tools::print_vector(suppc, "suppc");
   }

   // 1. load operators 
   using Tm = typename Km::dtype;
   oper_dict<Tm> lqops, rqops, cqops;
   oper_load_qops(icomb, p, scratch, "l", lqops, rank);
   oper_load_qops(icomb, p, scratch, "r", rqops, rank);
   oper_load_qops(icomb, p, scratch, "c", cqops, rank);
   if(rank == 0){
      std::cout << "qops info: rank=" << rank << std::endl;
      lqops.print("lqops");
      rqops.print("rqops");
      cqops.print("cqops");
      size_t tsize = lqops.size()+rqops.size()+cqops.size();
      std::cout << " optot=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
   }
   timing.ta = tools::get_time();

   // 2. onedot wavefunction
   //	  |
   //   --*--
   const auto& ql = lqops.qbra;
   const auto& qr = rqops.qbra;
   const auto& qc = cqops.qbra;
   auto sym_state = get_qsym_state(isym, schd.nelec, schd.twoms);
   stensor3<Tm> wf(sym_state, ql, qr, qc, dir_WF3);
   if(rank == 0) wf.print("wf"); 

   // 3. Davidson solver for wf
   int nsub = wf.size();
   int neig = sweeps.nroots;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);

   // 3.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   onedot_Hdiag(lqops, rqops, cqops, ecore, wf, diag, size, rank);
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
	   			       {"c",cqops}};
   HVec_type<Tm> HVec;
   Hx_functors<Tm> Hx_funs;
   symbolic_task<Tm> H_formulae;
   std::map<qsym,qinfo3<Tm>> info_dict;
   Tm* workspace;
   using std::placeholders::_1;
   using std::placeholders::_2;
   if(schd.ctns.alg_hvec == 0){
      Hx_funs = onedot_Hx_functors(lqops, rqops, cqops, 
		                   int2e, ecore, wf, size, rank);
      HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
           	  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec > 0){
      std::string fname;
      if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
	                                + "_"+std::to_string(isweep)
	                		+ "_"+std::to_string(ibond)+".txt"; 
      H_formulae = symbolic_onedot_formulae(lqops, rqops, cqops, 
		                            int2e, size, rank, fname);
      if(schd.ctns.alg_hvec == 1){
         HVec = bind(&ctns::symbolic_Hx<Tm,stensor3<Tm>>, _1, _2, std::cref(H_formulae),
           	     std::cref(qops_dict), std::cref(ecore),
                     std::ref(wf), std::cref(size), std::cref(rank));
      }else if(schd.ctns.alg_hvec == 2){
	 size_t opsize = preprocess_opsize(qops_dict);
	 size_t wfsize = preprocess_wf3size(wf.info, info_dict);
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
         HVec = bind(&ctns::symbolic_Hx2<Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, std::cref(H_formulae),
           	     std::cref(qops_dict), std::cref(ecore), 
                     std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		     std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		     std::ref(workspace));
      }
   }
   oper_timer.clear();
   onedot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, wf);
   if(schd.ctns.alg_hvec == 2) delete[] workspace; 
   timing.tc = tools::get_time();
   if(rank == 0){ 
      sweeps.print_eopt(isweep, ibond);
      if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
   }

   // 4. decimation & renormalize operators
   onedot_renorm(schd, sweeps, isweep, ibond, icomb, vsol, wf, 
		 lqops, rqops, cqops, int2e, int1e, scratch);

   timing.t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::sweep_onedot", timing.t0, timing.t1);
      timing.analysis();
      sweeps.timing_global.accumulate(timing);
   }
}

// use one dot algorithm to produce a final wavefunction
// in right canonical form (RCF) for later usage
template <typename Km>
void sweep_rwfuns(const input::schedule& schd,
		  comb<Km>& icomb,
		  const integral::two_body<typename Km::dtype>& int2e,
	          const integral::one_body<typename Km::dtype>& int1e,
		  const double ecore,
		  const std::string scratch){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::sweep_rwfuns" << std::endl;

   // perform an additional onedot opt  
   auto p0 = std::make_pair(0,0);
   auto p1 = std::make_pair(1,0);
   auto cturn = icomb.topo.is_cturn(p0,p1);
   auto dbond = directed_bond(p0,p1,0,p1,cturn); // fake dbond
   const int dcut1 = -1;
   const double eps = schd.ctns.ctrls[schd.ctns.maxsweep-1].eps; // take the last eps 
   input::params_sweep ctrl = {0, 1, dcut1, eps, 0.0};
   sweep_data sweeps({dbond}, schd.ctns.nroots, schd.ctns.guess, 
		      1, {ctrl}, 0, schd.ctns.rdm_vs_svd);
   sweep_onedot(schd, sweeps, 0, 0, icomb, int2e, int1e, ecore, scratch);

   if(rank == 0){
      std::cout << "deal with site0 by decimation for rsite0 & rwfuns" << std::endl;
      // decimation to get site0
      const auto& wf = icomb.psi[0]; // only rank-0 has psi from renorm
      stensor2<Tm> rot;
      int nroots = schd.ctns.nroots;
      std::vector<stensor2<Tm>> wfs2(nroots);
      for(int i=0; i<nroots; i++){
         auto wf2 = icomb.psi[i].merge_cr().T();
	 wfs2[i] = std::move(wf2);
      }
      const int dcut = nroots;
      double dwt; 
      int deff;
      const bool ifkr = tools::is_complex<Km>();
      std::string fname = scratch+"/decimation_site0.txt";
      decimation_row(ifkr, wf.info.qmid, wf.info.qcol, 
		     dcut, schd.ctns.rdm_vs_svd, wfs2,
		     rot, dwt, deff, fname);
      rot = rot.T(); 
      icomb.rsites[icomb.topo.rindex.at(p0)] = rot.split_cr(wf.info.qmid, wf.info.qcol);
      // form rwfuns(iroot,irbas)
      auto& sym_state = icomb.psi[0].info.sym;
      qbond qrow({{sym_state, nroots}});
      auto& qcol = rot.info.qrow; 
      stensor2<typename Km::dtype> rwfuns(qsym(Km::isym), qrow, qcol, {0,1});
      assert(qcol.size() == 1);
      int rdim = qrow.get_dim(0);
      int cdim = qcol.get_dim(0);
      for(int i=0; i<nroots; i++){
         auto cwf = icomb.psi[i].merge_cr().dot(rot.H()); // <-W[1,alpha]->
         for(int ic=0; ic<cdim; ic++){
            rwfuns(0,0)(i,ic) = cwf(0,0)(0,ic);
         }
      } // iroot
      icomb.rwfuns = std::move(rwfuns);
   } // rank0
}
   
} // ctns

#endif
