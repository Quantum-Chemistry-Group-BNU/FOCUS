#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_onedot_renorm.h"
#include "sweep_onedot_hdiag.h"
#include "sweep_onedot_local.h"
#include "sweep_onedot_sigma.h"
#include "symbolic_formulae_onedot.h"
#include "symbolic_preprocess.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"

namespace ctns{

// onedot optimization algorithm
template <typename Km>
void sweep_onedot(comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore,
		  const input::schedule& schd,
		  const std::string scratch,
	          oper_pool<typename Km::dtype>& qops_pool,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond){
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
      std::cout << "ctns::sweep_onedot"
	        << " alg_hvec=" << schd.ctns.alg_hvec
		<< " alg_renorm=" << schd.ctns.alg_renorm
                << " mpisize=" << size
	        << " maxthreads=" << maxthreads 
	        << std::endl;
   }
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // check partition 
   const auto& dbond = sweeps.seq[ibond];
   const bool ifNC = icomb.topo.check_partition(1, dbond, debug);

   // 1. load operators 
   auto fneed = icomb.topo.get_fqops(1, dbond, scratch, debug);
   qops_pool.fetch(fneed);
   const oper_dictmap<Tm> qops_dict = {{"l",qops_pool(fneed[0])},
	   		 	       {"r",qops_pool(fneed[1])},
	   			       {"c",qops_pool(fneed[2])}};
   if(debug){
      std::cout << "qops info: rank=" << rank << std::endl;
      qops_dict.at("l").print("lqops");
      qops_dict.at("r").print("rqops");
      qops_dict.at("c").print("cqops");
      size_t tsize = qops_dict.at("l").size()
	      	   + qops_dict.at("r").size()
		   + qops_dict.at("c").size();
      std::cout << " qops(tot)=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
   }
   timing.ta = tools::get_time();

   // 2. onedot wavefunction
   //	  |
   //   --*--
   const auto& ql = qops_dict.at("l").qket;
   const auto& qr = qops_dict.at("r").qket;
   const auto& qc = qops_dict.at("c").qket;
   auto sym_state = get_qsym_state(Km::isym, schd.nelec, schd.twoms);
   stensor3<Tm> wf(sym_state, ql, qr, qc, dir_WF3);
   if(debug) wf.print("wf"); 

   // 3. Davidson solver for wf
   int nsub = wf.size();
   int neig = sweeps.nroots;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);

   // 3.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   onedot_Hdiag(qops_dict, ecore, wf, diag, size, rank);
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
   std::map<qsym,qinfo3<Tm>> info_dict;
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
      Hx_funs = onedot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank);
      HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
           	  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 1){
      H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
					    schd.ctns.sort_formulae);
      HVec = bind(&ctns::symbolic_Hx<Tm,stensor3<Tm>>, _1, _2, std::cref(H_formulae),
        	  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 2){
      H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
					    schd.ctns.sort_formulae);
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx2<Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
		  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
     	          std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
     	          std::ref(workspace));
   }else if(schd.ctns.alg_hvec == 3){
      H_formulae2 = symbolic_formulae_onedot2(qops_dict, int2e, size, rank, fname,
					      schd.ctns.sort_formulae);
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx3<Tm,stensor3<Tm>,qinfo3<Tm>>, _1, _2, 
		  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
     	          std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
     	          std::ref(workspace));
   } // alg_hvec
   oper_timer.clear();
   onedot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, wf);
   if(preprocess) delete[] workspace;
   if(debug){
      sweeps.print_eopt(isweep, ibond);
      if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
   }
   timing.tc = tools::get_time();

   // 3. decimation & renormalize operators
   auto fbond = icomb.topo.get_fbond(dbond, scratch, debug);
   auto frop = fbond.first;
   auto fdel = fbond.second;
   onedot_renorm(icomb, int2e, int1e, schd, scratch, 
		 vsol, wf, qops_dict, qops_pool(frop), 
		 sweeps, isweep, ibond);
   timing.tf = tools::get_time();
   
   // 4. save on disk 
   qops_pool.save(frop);
   /*
      NOTE: At the boundary case [ -*=>=*-* and -*=<=*-* ],
      removing in the later configuration must wait until 
      the file from the former configuration has been saved!
      Therefore, oper_remove must come later than save,
      which contains the synchronization!
   */
   oper_remove(fdel, debug);

   timing.t1 = tools::get_time();
   if(debug){
      tools::timing("ctns::sweep_onedot", timing.t0, timing.t1);
      timing.analysis("time_local");
   }
}

// use one dot algorithm to produce a final wavefunction
// in right canonical form (RCF) for later usage
template <typename Km>
void sweep_rwfuns(comb<Km>& icomb,
		  const integral::two_body<typename Km::dtype>& int2e,
	          const integral::one_body<typename Km::dtype>& int1e,
		  const double ecore,
		  const input::schedule& schd,
		  const std::string scratch,
	          oper_pool<typename Km::dtype>& qops_pool){
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
   auto dbond = directed_bond(p0,p1,0); // fake dbond
   const int dcut1 = -1;
   const double eps = schd.ctns.ctrls[schd.ctns.maxsweep-1].eps; // take the last eps 
   input::params_sweep ctrl = {0, 1, dcut1, eps, 0.0};
   sweep_data sweeps({dbond}, schd.ctns.nroots, schd.ctns.guess, 
		      1, {ctrl}, 0, schd.ctns.rdm_vs_svd);
   sweep_onedot(icomb, int2e, int1e, ecore, schd, scratch,
		qops_pool, sweeps, 0, 0);

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
