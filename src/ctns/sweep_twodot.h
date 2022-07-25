#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_twodot_renorm.h"
#include "sweep_twodot_diag.h"
#include "sweep_twodot_local.h"
#include "sweep_twodot_sigma.h"
#include "symbolic_formulae_twodot.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"
#include "preprocess_size.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma2.h"
#include "preprocess_sigma_batch.h"

namespace ctns{

// twodot optimization algorithm
template <typename Km>
void sweep_twodot(comb<Km>& icomb,
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
   icomb.topo.check_partition(2, dbond, debug, schd.ctns.verbose);

   // 1. load operators 
   auto fneed = icomb.topo.get_fqops(2, dbond, scratch, debug && schd.ctns.verbose>0);
   qops_pool.fetch(fneed);
   const oper_dictmap<Tm> qops_dict = {{"l" ,qops_pool(fneed[0])},
	   		 	       {"r" ,qops_pool(fneed[1])},
	   			       {"c1",qops_pool(fneed[2])},
				       {"c2",qops_pool(fneed[3])}};
   if(debug && schd.ctns.verbose>0){
      std::cout << "qops info: rank=" << rank << std::endl;
      qops_dict.at("l").print("lqops");
      qops_dict.at("r").print("rqops");
      qops_dict.at("c1").print("c1qops");
      qops_dict.at("c2").print("c2qops");
      size_t tsize = qops_dict.at("l").size()
	           + qops_dict.at("r").size()
		   + qops_dict.at("c1").size()
		   + qops_dict.at("c2").size();
      std::cout << " qops(tot)=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
   }
   timing.ta = tools::get_time();

   // 2. twodot wavefunction
   //	 \ /
   //   --*--
   const auto& ql  = qops_dict.at("l").qket;
   const auto& qr  = qops_dict.at("r").qket;
   const auto& qc1 = qops_dict.at("c1").qket;
   const auto& qc2 = qops_dict.at("c2").qket;
   auto sym_state = get_qsym_state(Km::isym, schd.nelec, schd.twoms);
   stensor4<Tm> wf(sym_state, ql, qr, qc1, qc2);
   if(debug){
      std::cout << "wf4(diml,dimr,dimc1,dimc2)=(" 
	        << ql.get_dimAll() << ","
		<< qr.get_dimAll() << ","
		<< qc1.get_dimAll() << ","
		<< qc2.get_dimAll() << ")"
		<< " nnz=" << wf.size() << ":"
		<< tools::sizeMB<Tm>(wf.size()) << "MB"
	        << std::endl;
      if(schd.ctns.verbose>0) wf.print("wf");
   }
 
   // 3. Davidson solver for wf
   size_t ndim = wf.size();
   int neig = sweeps.nroots;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(ndim,neig);
   
   // 3.1 diag 
   std::vector<double> diag(ndim);
   twodot_diag(qops_dict, ecore, wf, diag, size, rank, schd.ctns.ifdist1);
#ifndef SERIAL
   // reduction of partial diag: no need to broadcast, if only rank=0 
   // executes the preconditioning in Davidson's algorithm
   if(size > 1){
      std::vector<double> diag2(ndim);
      boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
      diag = std::move(diag2);
   }
#endif 
   timing.tb = tools::get_time();

   // 3.2 Solve local problem: Hc=cE
   std::map<qsym,qinfo4<Tm>> info_dict;
   size_t opsize, wfsize, tmpsize, worktot;
   opsize = preprocess_opsize(qops_dict);
   wfsize = preprocess_wfsize(wf.info, info_dict);
   std::string fname;
   if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
	      			     + "_isweep"+std::to_string(isweep)
	                             + "_ibond"+std::to_string(ibond) + ".txt";
   HVec_type<Tm> HVec; 
   Hx_functors<Tm> Hx_funs; // hvec0
   symbolic_task<Tm> H_formulae; // hvec1,2
   bipart_task<Tm> H_formulae2; // hvec3
   intermediates<Tm> inter; // hvec4,5,6
   Hxlist<Tm> Hxlst; // hvec4
   Hxlist2<Tm> Hxlst2; // hvec5
   MMtasks<Tm> mmtasks; // hvec6
   Tm scale = Km::ifkr? 0.5*ecore : 1.0*ecore;
   std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
   Tm* opaddr[5] = {qops_dict.at("l")._data, qops_dict.at("r")._data,
	            qops_dict.at("c1")._data, qops_dict.at("c2")._data,
	 	    nullptr};
   size_t blksize;
   double cost;
   Tm* workspace;
   using std::placeholders::_1;
   using std::placeholders::_2;
   const bool debug_formulae = schd.ctns.verbose>0;
   if(tools::is_complex<Tm>() && schd.ctns.alg_hvec >=4){
      std::cout << "inter does not support cNK yet!" << std::endl;
      exit(1); 
   }
   if(schd.ctns.alg_hvec == 0){
      Hx_funs = twodot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
		                   schd.ctns.ifdist1, debug_formulae);
      HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 1){
      // raw version: symbolic formulae + dynamic allocation of memory 
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1,
					    debug_formulae); 
      HVec = bind(&ctns::symbolic_Hx<Tm,stensor4<Tm>>, _1, _2, std::cref(H_formulae),
        	  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.alg_hvec == 2){ 
      // symbolic formulae + preallocation of workspace 
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					    debug_formulae);
      tmpsize = opsize + 3*wfsize;
      worktot = maxthreads*tmpsize;
      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx2<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
		  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		  std::ref(workspace));
   }else if(schd.ctns.alg_hvec == 3){
      // symbolic formulae (factorized) + preallocation of workspace 
      H_formulae2 = symbolic_formulae_twodot2(qops_dict, int2e, size, rank, fname,
			                      schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					      debug_formulae); 
      tmpsize = opsize + 4*wfsize;
      worktot = maxthreads*tmpsize;
      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      workspace = new Tm[worktot];
      HVec = bind(&ctns::symbolic_Hx3<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
		  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
		  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
		  std::ref(workspace));
   }else if(schd.ctns.alg_hvec == 4){
      // Single Hxlst	   
      // symbolic formulae + intermediates + preallocation of workspace
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					    debug_formulae); 
      preprocess_formulae_sigma(qops_dict, oploc, H_formulae, wf, inter, 
		       		Hxlst, blksize, cost, schd.ctns.hxorder, 
				rank==0 && schd.ctns.verbose>0);
      if(schd.ctns.verbose>0){
         for(int k=0; k<size; k++){
            if(rank == k){
               if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
               std::cout << " * rank=" << k 
			 << " size(Hxlst)=" << Hxlst.size()
			 << " blksize=" << blksize
			 << " cost=" << cost 
			 << std::endl;
            }
            icomb.world.barrier();
         }
      }
      opaddr[4] = inter._data;
      worktot = maxthreads*(blksize*2+ndim);
      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      HVec = bind(&ctns::preprocess_Hx<Tm>, _1, _2,
		  std::cref(scale), std::cref(size), std::cref(rank),
		  std::cref(ndim), std::cref(blksize), 
		  std::ref(Hxlst), std::ref(opaddr));
   }else if(schd.ctns.alg_hvec == 5){
      // Hxlist2 
      // symbolic formulae + intermediates + preallocation of workspace
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					    debug_formulae); 
      preprocess_formulae_sigma2(qops_dict, oploc, H_formulae, wf, inter, 
		       		 Hxlst2, blksize, cost, schd.ctns.hxorder, 
		      		 rank==0 && schd.ctns.verbose>0);
      if(schd.ctns.verbose>0){
         for(int k=0; k<size; k++){
            if(rank == k){
               if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
	       size_t hxsize = 0;
      	       for(int i=0; i<Hxlst2.size(); i++){
      	          hxsize += Hxlst2[i].size();
      	       }
               std::cout << " * rank=" << k 
			 << " size(Hxlst)=" << hxsize 
			 << " blksize=" << blksize
			 << " cost=" << cost
			 << std::endl;
            }
            icomb.world.barrier();
         }
      }
      opaddr[4] = inter._data;
      worktot = maxthreads*blksize*3;
      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      HVec = bind(&ctns::preprocess_Hx2<Tm>, _1, _2,
		  std::cref(scale), std::cref(size), std::cref(rank),
		  std::cref(ndim), std::cref(blksize), 
		  std::ref(Hxlst2), std::ref(opaddr));
   }else if(schd.ctns.alg_hvec == 6){
      // BatchGEMM
      // symbolic formulae + intermediates + preallocation of workspace
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					    debug_formulae); 
      preprocess_formulae_sigma_batch(qops_dict, oploc, H_formulae, wf, inter, 
		       		      Hxlst2, blksize, cost, schd.ctns.hxorder,
				      mmtasks, schd.ctns.batchgemm, schd.ctns.batchsize,
		      		      rank==0 && schd.ctns.verbose>0);
      opaddr[4] = inter._data;
      worktot = mmtasks[0].batchsize*blksize*2;
      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      workspace = new Tm[worktot];
      HVec = bind(&ctns::preprocess_Hx_batch<Tm>, _1, _2,
		  std::cref(scale), std::cref(size), std::cref(rank),
		  std::cref(ndim), std::cref(blksize), 
		  std::ref(Hxlst2), std::ref(mmtasks), std::ref(opaddr), std::ref(workspace));
   }else if(schd.ctns.alg_hvec == 7){
/*
      // BatchGEMM on GPU
      // symbolic formulae + intermediates + preallocation of workspace
      H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
			                    schd.ctns.sort_formulae, schd.ctns.ifdist1, 
					    debug_formulae); 
      size_t blksize = preprocess_formulae_sigma_batch(qops_dict, oploc, H_formulae, wf, inter, 
		       			         Hxlst2, schd.ctns.hxorder,
						 mmtasks, schd.ctns.batchgemm, schd.ctns.batchsize,
		      		                 rank==0 && schd.ctns.verbose>0);
      opaddr[4] = inter._data;

      // GPU: copy operators (qops_dict & inter)
      // 1. allocate memery on GPU
      size_t tsize = qops_dict.at("l").size()
	           + qops_dict.at("r").size()
		   + qops_dict.at("c1").size()
		   + qops_dict.at("c2").size()
		   + inter.size();
      // 2. copy
      // qops_dict.at("l")._data;
      // inter._data
      // 3. save pointers to opaddr
      // opaddr[0] =  
      // opaddr[1] = 
      // opaddr[2] = 
      // opaddr[3] = 
      // opaddr[4] = 
      std::cout << " qops(tot)=" << tsize 
                << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                << ":" << tools::sizeGB<Tm>(tsize) << "GB"
		<< std::endl;
      // 4. allocate memory for Davidson: x,y(=Hx),worktot
      worktot = mmtasks[0].batchsize*blksize*2;
      workspace = new Tm[2*ndim+worktot];

      if(debug && schd.ctns.verbose>0){
         std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                   << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                   << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
      }
      workspace = new Tm[worktot];
      HVec = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
		  std::cref(scale), std::cref(size), std::cref(rank),
		  std::cref(ndim), std::cref(blksize), 
		  std::ref(Hxlst2), std::ref(mmtasks), std::ref(opaddr), std::ref(workspace));
*/
   }else{
      std::cout << "error: no such option for alg_hvec=" << schd.ctns.alg_hvec << std::endl;
      exit(1);
   } // alg_hvec
   oper_timer.clear();
   twodot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2,
		  ndim, neig, diag, HVec, eopt, vsol, nmvp, wf, dbond);
   // free temporary space
   if(schd.ctns.alg_hvec==2 || schd.ctns.alg_hvec==3 ||
      schd.ctns.alg_hvec==6 || schd.ctns.alg_hvec==7){
      delete[] workspace;
      // free memory space on GPU
      if(schd.ctns.alg_hvec == 7){}
   }
   if(debug && schd.ctns.verbose>1){
      sweeps.print_eopt(isweep, ibond);
      if(schd.ctns.alg_hvec == 0) oper_timer.analysis();
   }
   timing.tc = tools::get_time();

   // 3. decimation & renormalize operators
   auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
   auto frop = fbond.first;
   auto fdel = fbond.second;
   twodot_renorm(icomb, int2e, int1e, schd, scratch, 
		 vsol, wf, qops_dict, qops_pool(frop), 
		 sweeps, isweep, ibond);
   timing.tf = tools::get_time();
  
   // 4. save on disk 
   qops_pool.save(frop);
   oper_remove(fdel, debug);

   timing.t1 = tools::get_time();
   if(debug) timing.analysis("time_local", schd.ctns.verbose>0);
}

} // ctns

#endif
