#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "../qtensor/qtensor.h"
#include "sweep_util.h"
#include "sweep_onedot_renorm.h"
#include "sweep_onedot_diag.h"
#include "sweep_onedot_local.h"
#include "sweep_onedot_sigma.h"
#include "symbolic_formulae_onedot.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"
#include "preprocess_size.h"
#include "preprocess_hxlist.h"
#include "preprocess_hformulae.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma_batch.h"
#include "sadmrg/sweep_onedot_diag_su2.h"
//#include "sadmrg/symbolic_formulae_onedot_su2.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "preprocess_sigma_batchGPU.h"
#include "sweep_onedot_diagGPU.h"
#include "sadmrg/sweep_onedot_diagGPU_su2.h"
#endif

namespace ctns{

   // onedot optimization algorithm
   template <typename Qm, typename Tm>
      void sweep_onedot(comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            qoper_pool<Qm::ifabelian,Tm>& qops_pool,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const bool ifab = Qm::ifabelian;
         const int alg_hvec = schd.ctns.alg_hvec;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_onedot"
               << " ifabelian=" << ifab
               << " alg_hvec=" << alg_hvec
               << " alg_renorm=" << alg_renorm
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
         }
         auto& timing = sweeps.opt_timing[isweep][ibond];
         timing.t0 = tools::get_time();

         // check partition 
         const auto& dbond = sweeps.seq[ibond];
         icomb.topo.check_partition(1, dbond, debug, schd.ctns.verbose);

         // 1. load operators 
         auto fneed = icomb.topo.get_fqops(1, dbond, scratch, debug && schd.ctns.verbose>0);
         qops_pool.fetch_to_memory(fneed, alg_hvec>10 || alg_renorm>10);
         const qoper_dictmap<ifab,Tm> qops_dict = {
            {"l",qops_pool.at(fneed[0])},
            {"r",qops_pool.at(fneed[1])},
            {"c",qops_pool.at(fneed[2])}
         };
         size_t opertot = qops_dict.at("l").size()
            + qops_dict.at("r").size()
            + qops_dict.at("c").size();
         if(debug && schd.ctns.verbose>0){
            std::cout << "qops info: rank=" << rank << std::endl;
            qops_dict.at("l").print("lqops");
            qops_dict.at("r").print("rqops");
            qops_dict.at("c").print("cqops");
            std::cout << " qops(tot)=" << opertot
               << ":" << tools::sizeMB<Tm>(opertot) << "MB"
               << ":" << tools::sizeGB<Tm>(opertot) << "GB"
               << std::endl;
         }
         timing.ta = tools::get_time();

         // 1.5 look ahead for the next dbond
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
         /*
         auto frop = fbond.first;
         auto fdel = fbond.second;
         auto fneed_next = sweep_fneed_next(icomb, scratch, sweeps, isweep, ibond, debug && schd.ctns.verbose>0);
         // prefetch files for the next bond
         if(schd.ctns.async_fetch){
            if(alg_hvec>10 && alg_renorm>10){
               const bool ifkeepcoper = schd.ctns.alg_hcoper>=1 || schd.ctns.alg_rcoper>=1;
               qops_pool.clear_from_cpumem(fneed, fneed_next, ifkeepcoper);
            }
            qops_pool[frop]; // just declare a space for frop
            qops_pool.fetch_to_cpumem(fneed_next, schd.ctns.async_fetch); // just to cpu
         }
         timing.ta = tools::get_time();
         */

         // 2. onedot wavefunction
         //	    |
         //   --*--
         const auto& ql = qops_dict.at("l").qket;
         const auto& qr = qops_dict.at("r").qket;
         const auto& qc = qops_dict.at("c").qket;
         auto sym_state = get_qsym_state(Qm::isym, schd.nelec, 
               (ifab? schd.twoms : schd.twos),
               schd.ctns.singlet);
         qtensor3<ifab,Tm> wf(sym_state, ql, qr, qc, dir_WF3); // su2 case: by default, CRcouple is used.
         size_t ndim = wf.size();
         int neig = sweeps.nroots;
         if(debug){
            std::cout << "wf3(diml,dimr,dimc)=(" 
               << ql.get_dimAll() << ","
               << qr.get_dimAll() << ","
               << qc.get_dimAll() << ")"
               << " nnz=" << ndim  << ":"
               << tools::sizeMB<Tm>(ndim) << "MB"
               << std::endl;
            wf.print("wf3",schd.ctns.verbose-2);
         }
         if(ndim == 0){
            std::cout << "error: symmetry is inconsistent as ndim=0" << std::endl;
            exit(1);
         }
         if(ndim < neig){
            std::cout << "error: ndim<neig! either neig is too large or dcut is too small." << std::endl;
            exit(1);
         }

         // 3. Davidson solver for wf
         // 3.1 diag 
         double* diag = new double[ndim];
         if(alg_hvec <= 10){
            onedot_diag(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1);
#ifdef GPU
         }else{
            onedot_diagGPU(qops_dict, wf, diag, size, rank, schd.ctns.ifdist1, schd.ctns.ifnccl, schd.ctns.diagcheck);
#endif
         }
#ifndef SERIAL
         // reduction of partial diag: no need to broadcast, if only rank=0 
         // executes the preconditioning in Davidson's algorithm
         if(size > 1){
            mpi_wrapper::reduce(icomb.world, diag, ndim, 0);
         }
#endif 
         std::transform(diag, diag+ndim, diag,
               [&ecore](const double& x){ return x+ecore; });
         timing.tb = tools::get_time();

         // 3.2 Solve local problem: Hc=cE
         // prepare HVec
         std::map<qsym,qinfo3type<ifab,Tm>> info_dict;
         size_t opsize=0, wfsize=0, tmpsize=0, worktot=0;
         opsize = preprocess_opsize<ifab,Tm>(qops_dict);
         wfsize = preprocess_wfsize<ifab,Tm>(wf.info, info_dict);
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/hformulae"
            + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         HVec_type<Tm> HVec;
         Hx_functors<Tm> Hx_funs; // hvec0
         symbolic_task<Tm> H_formulae; // hvec1,2
         bipart_task<Tm> H_formulae2; // hvec3
         hintermediates<ifab,Tm> hinter; // hvec4,5,6
         Hxlist<Tm> Hxlst; // hvec4
         Hxlist2<Tm> Hxlst2; // hvec5
         HMMtask<Tm> Hmmtask;
         HMMtasks<Tm> Hmmtasks; // hvec6
         Tm scale = qkind::is_qNK<Qm>()? 0.5*ecore : 1.0*ecore;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {qops_dict.at("l")._data, qops_dict.at("r")._data, qops_dict.at("c")._data, 
            nullptr, nullptr};
         size_t blksize=0, blksize0=0;
         double cost=0.0;
         Tm* workspace = nullptr;
#ifdef GPU
         Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
         Tm* dev_workspace = nullptr;
         Tm* dev_red = nullptr;
#endif
         size_t batchsize=0, gpumem_dvdson=0, gpumem_batch=0;

         using std::placeholders::_1;
         using std::placeholders::_2;
         const bool debug_formulae = schd.ctns.verbose>0;
         std::string fmmtask;
         if(debug && schd.ctns.save_mmtask && isweep == schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond){
            fmmtask = "hmmtasks_isweep"+std::to_string(isweep) + "_ibond"+std::to_string(ibond);
         }

         // consistency check
         if(schd.ctns.ifdistc && !icomb.topo.ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }
         if(ifab && Qm::ifkr && alg_hvec >=4){
            std::cout << "error: alg_hvec >=4 does not support onedot yet! GEMM with conj is needed." << std::endl;
            exit(1); 
         }
         if(alg_hvec < 10 && schd.ctns.alg_hinter == 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter<2" << std::endl;
            exit(1);
         }
         if(alg_hvec > 10 && schd.ctns.alg_hinter != 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter=2" << std::endl;
            exit(1);
         }

         timing.tb1 = tools::get_time();
         if(alg_hvec == 0){

            // oldest version
            Hx_funs = onedot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
                  schd.ctns.ifdist1, debug_formulae);
            HVec = bind(&ctns::onedot_Hx<ifab,Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            HVec = bind(&ctns::symbolic_Hx<ifab,Tm,qtensor3<ifab,Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 2){

            // symbolic formulae + preallocation of workspace 
            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            HVec = bind(&ctns::symbolic_Hx2<ifab,Tm,qtensor3<ifab,Tm>,qinfo3type<ifab,Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            H_formulae2 = symbolic_formulae_onedot2(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            HVec = bind(&ctns::symbolic_Hx3<ifab,Tm,qtensor3<ifab,Tm>,qinfo3type<ifab,Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 4){

            // OpenMP + Single Hxlst: symbolic formulae + hintermediates + preallocation of workspace

            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist2(Hxlst);

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

         }else if(alg_hvec == 5){

            // OpenMP + Hxlist2: symbolic formulae + hintermediates + preallocation of workspace

            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae); 

            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist2(Hxlst2);

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

         }else if(alg_hvec == 6 || alg_hvec == 7 || alg_hvec == 8 || alg_hvec == 9){

            // BatchGEMM: symbolic formulae + hintermediates + preallocation of workspace

            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            const bool ifSingle = alg_hvec > 7;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = 1;
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, debug);

            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                     Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Hxlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Hxlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                     Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Hxlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 

            if(blksize > 0){
               // determine batchsize dynamically
               size_t blocksize = 2*blksize+blksize0;
               preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, 
                     batchsize, worktot);
               if(debug && schd.ctns.verbose>0){
                  std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                     << " blksize0=" << blksize0 << " batchsize=" << batchsize
                     << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                     << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
               }
               workspace = new Tm[worktot];

               // generate Hmmtasks
               const int batchblas = schd.ctns.alg_hinter; // use the same keyword for GEMM_batch
               auto batchhvec = std::make_tuple(batchblas,batchblas,batchblas);
               if(!ifSingle){
                  Hmmtasks.resize(Hxlst2.size());
                  for(int i=0; i<Hmmtasks.size(); i++){
                     Hmmtasks[i].init(Hxlst2[i], schd.ctns.alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
                     if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                        std::cout << " rank=" << rank << " iblk=" << i 
                           << " size=" << Hxlst2[i][0].size 
                           << " Hmmtasks.totsize=" << Hmmtasks[i].totsize
                           << " batchsize=" << Hmmtasks[i].batchsize 
                           << " nbatch=" << Hmmtasks[i].nbatch 
                           << std::endl;
                     }
                  } // i
                  if(fmmtask.size()>0) save_mmtask(Hmmtasks, fmmtask);
               }else{
                  Hmmtask.init(Hxlst, schd.ctns.alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1){
                     std::cout << " rank=" << rank 
                        << " Hxlst.size=" << Hxlst.size()
                        << " Hmmtask.totsize=" << Hmmtask.totsize
                        << " batchsize=" << Hmmtask.batchsize 
                        << " nbatch=" << Hmmtask.nbatch 
                        << std::endl;
                  }
                  if(fmmtask.size()>0) save_mmtask(Hmmtask, fmmtask);
               }
            } // blksize>0

            if(!ifSingle){
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batch<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirect<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }else{
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }

#ifdef GPU
         }else if(alg_hvec == 16 || alg_hvec == 17 || alg_hvec == 18 || alg_hvec == 19){

            // BatchGEMM on GPU: symbolic formulae + hintermediates + preallocation of workspace

            // allocate memery on GPU & copy qops
            for(int i=0; i<3; i++){
               const auto& tqops = qops_pool.at(fneed[i]);
               assert(tqops.avail_gpu());
               dev_opaddr[i] = tqops._dev_data;
            }
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb2 = tools::get_time();

            H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, schd.ctns.ifdistc, debug_formulae);

            timing.tb3 = tools::get_time();

            // compute hintermediates on GPU directly
            const bool ifSingle = alg_hvec > 17;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = std::get<0>(schd.ctns.batchhvec); 
            hinter.init(ifDirect, schd.ctns.alg_hinter, batchgemv, qops_dict, oploc, dev_opaddr, H_formulae, debug);
            size_t gpumem_hinter = sizeof(Tm)*hinter.size();
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb4 = tools::get_time();
            timing.tb5 = tools::get_time();

            // GEMM list and GEMV list
            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Hxlist2(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter, 
                     Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Hxlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Hxlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Hxlist(ifDirect, schd.ctns.alg_hcoper, 
                     qops_dict, oploc, opaddr, H_formulae, wf, hinter, 
                     Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Hxlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 
            timing.tb6 = tools::get_time();

            // Determine batchsize dynamically
            gpumem_dvdson = sizeof(Tm)*2*ndim;
            if(blksize > 0){
               size_t blocksize = 2*blksize+blksize0+1;
               preprocess_gpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, gpumem_dvdson, rank,
                     batchsize, gpumem_batch);
            }
            dev_workspace = (Tm*)GPUmem.allocate(gpumem_dvdson+gpumem_batch);
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter,dvdson,batch)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << "," << gpumem_dvdson/std::pow(1024.0,3)
                  << "," << gpumem_batch/std::pow(1024.0,3)
                  << " blksize=" << blksize
                  << " blksize0=" << blksize0
                  << " batchsize=" << batchsize 
                  << std::endl;
            }

            // generate Hmmtasks given batchsize
            const int batchblas = 2; // GPU
            if(!ifSingle){
               Hmmtasks.resize(Hxlst2.size());
               for(int i=0; i<Hmmtasks.size(); i++){
                  Hmmtasks[i].init(Hxlst2[i], schd.ctns.alg_hcoper, batchblas, schd.ctns.batchhvec, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                     std::cout << " rank=" << rank << " iblk=" << i 
                        << " size=" << Hxlst2[i][0].size 
                        << " Hmmtasks.totsize=" << Hmmtasks[i].totsize
                        << " batchsize=" << Hmmtasks[i].batchsize 
                        << " nbatch=" << Hmmtasks[i].nbatch 
                        << std::endl;
                  }
               } // i
               if(fmmtask.size()>0) save_mmtask(Hmmtasks, fmmtask);
            }else{
               Hmmtask.init(Hxlst, schd.ctns.alg_hcoper, batchblas, schd.ctns.batchhvec, batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1){
                  std::cout << " rank=" << rank
                     << " Hxlst.size=" << Hxlst.size()
                     << " Hmmtasks.totsize=" << Hmmtask.totsize
                     << " batchsize=" << Hmmtask.batchsize 
                     << " nbatch=" << Hmmtask.nbatch 
                     << std::endl;
                  if(schd.ctns.verbose>2){
                     for(int k=0; k<Hmmtask.nbatch; k++){
                        for(int i=0; i<Hmmtask.mmbatch2[k].size(); i++){
                           if(Hmmtask.mmbatch2[k][i].size==0) continue;
                           std::cout << " Hmmbatch2: k/nbatch=" << k << "/" << Hmmtask.nbatch
                              << " i=" << i << " size=" << Hmmtask.mmbatch2[k][i].size
                              << " group=" << Hmmtask.mmbatch2[k][i].gsta.size()-1
                              << " average=" << Hmmtask.mmbatch2[k][i].size/(Hmmtask.mmbatch2[k][i].gsta.size()-1)
                              << std::endl;
                        }
                     }
                  }
               }
               if(fmmtask.size()>0) save_mmtask(Hmmtask, fmmtask);
            }
            timing.tb7 = tools::get_time();

            // GPU version of Hx
            dev_red = dev_workspace + 2*ndim + batchsize*(blksize*2+blksize0); 
            if(!ifSingle){
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(hinter._dev_data), std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }
            }else{
               if(!ifDirect){
                  HVec = bind(&ctns::preprocess_Hx_batchGPUSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  HVec = bind(&ctns::preprocess_Hx_batchDirectGPUSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(hinter._dev_data), std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }
            }

#endif // GPU

         }else{
            std::cout << "error: no such option for alg_hvec=" << alg_hvec << std::endl;
            exit(1);
         } // alg_hvec

         // solve HC=CE
         linalg::matrix<Tm> vsol(ndim,neig);
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.dot_start();
         onedot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2, 
               ndim, neig, diag, HVec, eopt, vsol, nmvp, wf, timing);
         if(debug){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
         }
         timing.tc = tools::get_time();

         // free temporary space
         delete[] diag;
         if(alg_hvec >=2){
            delete[] workspace;
         }

         // 3. decimation & renormalize operators
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         onedot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_pool, fneed, fneed_next, frop,
               sweeps, isweep, ibond);
         timing.tf = tools::get_time();

         // 4. save on disk
         if(debug){
            get_sys_status();
            icomb.display_size();
         }
         auto t0 = tools::get_time();
         qops_pool.join_and_erase(fneed, fneed_next); 
         auto t1 = tools::get_time();
         qops_pool.save_to_disk(frop, schd.ctns.async_save, 
               schd.ctns.alg_renorm>10 && schd.ctns.async_tocpu, fneed_next);
         auto t2 = tools::get_time();
         qops_pool.remove_from_disk(fdel, schd.ctns.async_remove);
         auto t3 = tools::get_time();
         if(debug){
            std::cout << "TIMING FOR cleanup: " << tools::get_duration(t3-t0)
               << " T(join&erase/save/remove)="
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << ","
               << tools::get_duration(t3-t2)
               << std::endl;
         }

         // save for restart
         if(rank == 0 && schd.ctns.timestamp) sweep_save(icomb, schd, scratch, sweeps, isweep, ibond);

         timing.t1 = tools::get_time();
         if(debug){
            get_sys_status();
            timing.analysis("local", schd.ctns.verbose>0);
         }
      }

} // ctns

#endif
