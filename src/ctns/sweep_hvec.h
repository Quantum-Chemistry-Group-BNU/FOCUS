#ifndef SWEEP_HVEC_H
#define SWEEP_HVEC_H

#include "../qtensor/qtensor.h"
#include "sweep_twodot_sigma.h"
#include "symbolic_formulae_twodot.h"
#include "symbolic_formulae_twodot2.h"
#include "sweep_onedot_sigma.h"
#include "symbolic_formulae_onedot.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"
#include "preprocess_size.h"
#include "preprocess_hformulae.h"
#include "preprocess_hxlist.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma_batch.h"
#ifdef GPU
#include "preprocess_sigma_batchGPU.h"
#endif

namespace ctns{

   template <typename Qm, typename Tm, typename QInfo, typename QTm>
      struct HVec_wrapper{
         public:
            void init(const int dots, 
                  const bool ifmps,
                  const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
                  const integral::two_body<Tm>& int2e,
                  const double& ecore,
                  const input::schedule& schd,
                  const int& size,
                  const int& rank,
                  const int& maxthreads,
                  const size_t& ndim,
                  QTm& wf,
                  dot_timing& timing,
                  const std::string& fname,
                  const std::string& fmmtask);
            // GEMM and GEMV list
            void init_Hxlist(const bool ifSingle,
                  const bool ifDirect,
                  const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
                  const QTm& wf,
                  const bool debug);
            // generate Hmmtasks
            void init_Hmmtask(const bool ifSingle,
                  const int& batchblas,
                  const std::tuple<int,int,int>& batchhvec,
                  const std::string& fmmtask,
                  const int rank,
                  const int verbose);
            // init Hx_functors
            template <typename QTm2=QTm, std::enable_if_t<std::is_same<QTm2,qtensor3<Qm::ifabelian,Tm>>::value,bool> = 0>
               void init_Hx_functors(const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
                     const integral::two_body<Tm>& int2e,
                     const double& ecore,
                     const QTm& wf,
                     const int& size,
                     const int& rank,
                     const bool& ifdist1,
                     const bool debug=false){
                  Hx_funs = onedot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, ifdist1, debug);
               }
            template <typename QTm2=QTm, std::enable_if_t<std::is_same<QTm2,qtensor4<Qm::ifabelian,Tm>>::value,bool> = 0>
               void init_Hx_functors(const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
                     const integral::two_body<Tm>& int2e,
                     const double& ecore,
                     const QTm& wf,
                     const int& size,
                     const int& rank,
                     const bool& ifdist1,
                     const bool debug=false){
                  Hx_funs = twodot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, ifdist1, debug);
               }
            // cleanup
            void finalize();
         public:
            HVec_type<Tm> Hx;
         private:
            int alg_hvec;
            int alg_hinter;
            int alg_hcoper;
            std::map<std::string,int> oploc; // l,r,c1,c2,inter
            Tm* opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
            Hx_functors<Tm> Hx_funs; // hvec0
            symbolic_task<Tm> H_formulae; // hvec1,2
            std::map<qsym,QInfo> info_dict; // hvec2
            size_t opsize=0, wfsize=0, tmpsize=0, worktot=0;
            bipart_task<Tm> H_formulae2; // hvec3
            hintermediates<Qm::ifabelian,Tm> hinter; // hvec4,5,6
            Hxlist<Tm> Hxlst; // hvec4
            Hxlist2<Tm> Hxlst2; // hvec5
            HMMtask<Tm> Hmmtask; // hvec6-9,16-19
            HMMtasks<Tm> Hmmtasks; 
            Tm scale;
            size_t blksize=0, blksize0=0;
            double cost=0.0;
            Tm* workspace = nullptr;
#ifdef GPU
            Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
            Tm* dev_workspace = nullptr;
            Tm* dev_red = nullptr;
#endif
            size_t maxbatch=0, batchsize=0, gpumem_dvdson=0, gpumem_batch=0;
      };

   template <bool ifab, typename Tm, typename QTm, std::enable_if_t<std::is_same<QTm,qtensor3<ifab,Tm>>::value,bool> = 0>
      void oldest_Hx(Tm* y,
            const Tm* x,
            Hx_functors<Tm>& Hx_funs,
            QTm& wf,
            const int size,
            const int rank){
         onedot_Hx(y, x, Hx_funs, wf, size, rank);
      }
   template <bool ifab, typename Tm, typename QTm, std::enable_if_t<std::is_same<QTm,qtensor4<ifab,Tm>>::value,bool> = 0>
      void oldest_Hx(Tm* y,
            const Tm* x,
            Hx_functors<Tm>& Hx_funs,
            QTm& wf,
            const int size,
            const int rank){
         twodot_Hx(y, x, Hx_funs, wf, size, rank);
      }

   template <typename Qm, typename Tm, typename QInfo, typename QTm>
      void HVec_wrapper<Qm,Tm,QInfo,QTm>::init(const int dots,
            const bool ifmps, 
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const double& ecore,
            const input::schedule& schd,
            const int& size,
            const int& rank,
            const int& maxthreads,
            const size_t& ndim,
            QTm& wf,
            dot_timing& timing,
            const std::string& fname,
            const std::string& fmmtask){
         // basic setup
         const bool ifab = Qm::ifabelian;
         alg_hvec = schd.ctns.alg_hvec;
         alg_hinter = schd.ctns.alg_hinter;
         alg_hcoper = schd.ctns.alg_hcoper;
         // consistency check
         if(schd.ctns.ifdistc && !ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }
         if(!ifab && alg_hvec < 4){
            std::cout << "error: use alg_hvec >= 4 for non-Abelian case! alg_hvec=" << alg_hvec << std::endl;
            exit(1);
         }
         if(ifab && Qm::ifkr && alg_hvec >=4){
            // This needs to be checked more carefully, for instance, complex case with rNS/cNS.
            std::cout << "error: alg_hvec >= 4 does not support complex yet! GEMM with conj is needed." << std::endl;
            exit(1); 
         }
         if(alg_hvec <= 10 && alg_hinter == 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter<2" << std::endl;
            exit(1);
         }
         if(alg_hvec > 10 && alg_hinter != 2){
            std::cout << "error: alg_hvec=" << alg_hvec << " should be used with alg_hinter=2" << std::endl;
            exit(1);
         }
         opaddr[0] = qops_dict.at("l")._data;
         opaddr[1] = qops_dict.at("r")._data;
         if(dots == 1){
            opaddr[2] = qops_dict.at("c")._data;
            oploc = {{"l",0},{"r",1},{"c",2}};
         }else{
            opaddr[2] = qops_dict.at("c1")._data;
            opaddr[3] = qops_dict.at("c2")._data;
            oploc = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
         }
         opsize = preprocess_opsize<ifab,Tm>(qops_dict);
         wfsize = preprocess_wfsize<ifab,Tm>(wf.info, info_dict);
         scale = qkind::is_qNK<Qm>()? 0.5*ecore : 1.0*ecore;
         using std::placeholders::_1;
         using std::placeholders::_2;

         // setup formulae
         timing.tb1 = tools::get_time();
         if(alg_hvec > 0){
            if(alg_hvec != 3){
               if(dots == 1){
                  H_formulae = symbolic_formulae_onedot(qops_dict, int2e, size, rank, fname,
                        schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                        schd.ctns.ifdistc, schd.ctns.verbose>0);
               }else{
                  H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                        schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                        schd.ctns.ifdistc, schd.ctns.verbose>0);
               }
            }else{
               if(dots == 1){
                  H_formulae2 = symbolic_formulae_onedot2(qops_dict, int2e, size, rank, fname,
                        schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                        schd.ctns.ifdistc, schd.ctns.verbose>0);
               }else{ 
                  H_formulae2 = symbolic_formulae_twodot2(qops_dict, int2e, size, rank, fname,
                        schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                        schd.ctns.ifdistc, schd.ctns.verbose>0); 
               }
            }
         }
         timing.tb2 = tools::get_time();

         // setup HVec 
         if(alg_hvec == 0){

            // oldest version
            this->init_Hx_functors(qops_dict, int2e, ecore, wf, size, rank,
                  schd.ctns.ifdist1, schd.ctns.verbose>0);

            Hx = bind(&ctns::oldest_Hx<ifab,Tm,QTm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            Hx = bind(&ctns::symbolic_Hx<ifab,Tm,QTm>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 2){ 

            // symbolic formulae + preallocation of workspace 
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];

            Hx = bind(&ctns::symbolic_Hx2<ifab,Tm,QInfo,QTm>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];

            Hx = bind(&ctns::symbolic_Hx3<ifab,Tm,QInfo,QTm>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 4){

            // OpenMP + Hxlst[single]: symbolic formulae + hintermediates + preallocation of workspace
            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, rank==0);
            timing.tb3 = tools::get_time();

            preprocess_formulae_Hxlist(ifDirect, alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            get_MMlist2(Hxlst);
            timing.tb4 = tools::get_time();
            timing.tb5 = tools::get_time();

            worktot = maxthreads*blksize*2;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            Hx = bind(&ctns::preprocess_Hx<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst), std::ref(opaddr));

         }else if(alg_hvec == 5){

            // OpenMP + Hxlist2: symbolic formulae + hintermediates + preallocation of workspace
            const bool ifDirect = false;
            const int batchgemv = 1;
            hinter.init(ifDirect, alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, rank==0);
            timing.tb3 = tools::get_time();

            preprocess_formulae_Hxlist2(ifDirect, alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            get_MMlist2(Hxlst2);
            timing.tb4 = tools::get_time();
            timing.tb5 = tools::get_time();

            worktot = maxthreads*blksize*3;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            Hx = bind(&ctns::preprocess_Hx2<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst2), std::ref(opaddr));

         }else if(alg_hvec == 6 || alg_hvec == 7 || alg_hvec == 8 || alg_hvec == 9){

            // BatchGEMM: symbolic formulae + hintermediates + preallocation of workspace
            const bool ifSingle = alg_hvec > 7;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = 1;
            hinter.init(ifDirect, alg_hinter, batchgemv, qops_dict, oploc, opaddr, H_formulae, rank==0);
            timing.tb3 = tools::get_time();

            // GEMM and GEMV list
            this->init_Hxlist(ifSingle, ifDirect, qops_dict, wf, rank==0 && schd.ctns.verbose>0);
            timing.tb4 = tools::get_time();

            if(blksize > 0){
               // determine batchsize dynamically
               size_t blocksize = 2*blksize+blksize0;
               preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, 
                     batchsize, worktot);
               if(rank==0 && schd.ctns.verbose>0){
                  std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                     << " blksize0=" << blksize0 << " batchsize=" << batchsize
                     << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                     << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
               }
               workspace = new Tm[worktot];

               // generate Hmmtasks
               const int batchblas = alg_hinter; // use the same keyword for GEMM_batch
               auto batchhvec = std::make_tuple(batchblas,batchblas,batchblas);
               this->init_Hmmtask(ifSingle, batchblas, batchhvec, fmmtask, rank, schd.ctns.verbose);
            } // blksize>0
            timing.tb5 = tools::get_time();

            if(!ifSingle){
               if(!ifDirect){
                  Hx = bind(&ctns::preprocess_Hx_batch<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  Hx = bind(&ctns::preprocess_Hx_batchDirect<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }else{
               if(!ifDirect){
                  Hx = bind(&ctns::preprocess_Hx_batchSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace));
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  Hx = bind(&ctns::preprocess_Hx_batchDirectSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(opaddr), std::ref(workspace),
                        std::ref(hinter._data));
               }
            }

#ifdef GPU
         }else if(alg_hvec == 16 || alg_hvec == 17 || alg_hvec == 18 || alg_hvec == 19){

            // BatchGEMM on GPU: symbolic formulae + hintermediates + preallocation of workspace

            // setup addresses for GPU
            size_t opertot = 0;
            for(const auto& pr : qops_dict){
               const auto& key = pr.first;
               const auto& qops = pr.second;
               assert(qops.avail_gpu());
               dev_opaddr[oploc.at(key)] = qops._dev_data;
               opertot += qops.size();
            }
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }

            // compute hintermediates on GPU directly
            const bool ifSingle = alg_hvec > 17;
            const bool ifDirect = alg_hvec % 2 == 1;
            const int batchgemv = std::get<0>(schd.ctns.batchhvec); 
            hinter.init(ifDirect, alg_hinter, batchgemv, qops_dict, oploc, dev_opaddr, H_formulae, rank==0);
            size_t gpumem_hinter = sizeof(Tm)*hinter.size();
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb3 = tools::get_time();

            // GEMM and GEMV list
            this->init_Hxlist(ifSingle, ifDirect, qops_dict, wf, rank==0 && schd.ctns.verbose>0);
            timing.tb4 = tools::get_time();

            // Determine batchsize dynamically
            gpumem_dvdson = sizeof(Tm)*2*ndim;
            size_t blocksize = 2*blksize+blksize0+1;
            if(blksize > 0){
               preprocess_gpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, gpumem_dvdson, rank,
                     batchsize, gpumem_batch);
            }
            if(rank==0 && schd.ctns.verbose>0){
               size_t used = GPUmem.used();
               size_t avail = GPUmem.available(rank);
               size_t total = used + avail;
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << used/std::pow(1024.0,3)
                  << " avail=" << avail/std::pow(1024.0,3) 
                  << " total=" << total/std::pow(1024.0,3) 
                  << " dvdson[need]=" << gpumem_dvdson/std::pow(1024.0,3)
                  << " batch[need]=" << gpumem_batch/std::pow(1024.0,3)
                  << " blksize=" << blksize
                  << " blksize0=" << blksize0
                  << " batchsize=" << batchsize 
                  << std::endl;
            }
            dev_workspace = (Tm*)GPUmem.allocate(gpumem_dvdson+gpumem_batch);
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter,dvdson,batch)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_hinter/std::pow(1024.0,3) 
                  << "," << gpumem_dvdson/std::pow(1024.0,3)
                  << "," << gpumem_batch/std::pow(1024.0,3)
                  << std::endl;
            }

            // generate Hmmtasks given batchsize
            const int batchblas = 2; // GPU
            this->init_Hmmtask(ifSingle, batchblas, schd.ctns.batchhvec, fmmtask, rank, schd.ctns.verbose);
            timing.tb5 = tools::get_time();

            // GPU version of Hx
            dev_red = dev_workspace + 2*ndim + batchsize*(blksize*2+blksize0); 
            if(!ifSingle){
               if(!ifDirect){
                  Hx = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  Hx = bind(&ctns::preprocess_Hx_batchDirectGPU<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(hinter._dev_data), std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }
            }else{
               if(!ifDirect){
                  Hx = bind(&ctns::preprocess_Hx_batchGPUSingle<Tm>, _1, _2,
                        std::cref(scale), std::cref(size), std::cref(rank),
                        std::cref(ndim), std::ref(Hmmtask), std::ref(dev_opaddr), std::ref(dev_workspace),
                        std::ref(dev_red), std::cref(schd.ctns.ifnccl));
               }else{
                  dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*blksize*2; // memory layout [workspace|inter]
                  Hx = bind(&ctns::preprocess_Hx_batchDirectGPUSingle<Tm>, _1, _2,
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
      }

   // finalize
   template <typename Qm, typename Tm, typename QInfo, typename QTm>
      void HVec_wrapper<Qm,Tm,QInfo,QTm>::finalize(){
         if(alg_hvec==2 || alg_hvec==3 || 
               alg_hvec==6 || alg_hvec==7 ||
               alg_hvec==8 || alg_hvec==9){
            delete[] workspace;
         }
#ifdef GPU
         if(alg_hvec>10) GPUmem.deallocate(dev_workspace, gpumem_dvdson+gpumem_batch);
#endif
      }

   // GEMM and GEMV list
   template <typename Qm, typename Tm, typename QInfo, typename QTm>
      void HVec_wrapper<Qm,Tm,QInfo,QTm>::init_Hxlist(const bool ifSingle,
            const bool ifDirect,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
            const QTm& wf,
            const bool debug){
         if(!ifSingle){
            preprocess_formulae_Hxlist2(ifDirect, alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst2, blksize, blksize0, cost, debug);
            for(int i=0; i<Hxlst2.size(); i++){
               maxbatch = std::max(maxbatch, Hxlst2[i].size());
            } // i
         }else{
            preprocess_formulae_Hxlist(ifDirect, alg_hcoper, 
                  qops_dict, oploc, opaddr, H_formulae, wf, hinter,
                  Hxlst, blksize, blksize0, cost, debug);
            maxbatch = Hxlst.size();
         }
         if(!ifDirect) assert(blksize0 == 0);
      } 

   // generate Hmmtasks
   template <typename Qm, typename Tm, typename QInfo, typename QTm>
      void HVec_wrapper<Qm,Tm,QInfo,QTm>::init_Hmmtask(const bool ifSingle,
            const int& batchblas,
            const std::tuple<int,int,int>& batchhvec,
            const std::string& fmmtask,
            const int rank,
            const int verbose){
         if(!ifSingle){
            Hmmtasks.resize(Hxlst2.size());
            for(int i=0; i<Hmmtasks.size(); i++){
               Hmmtasks[i].init(Hxlst2[i], alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
               if(rank==0 && verbose>1 && Hxlst2[i].size()>0){
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
            Hmmtask.init(Hxlst, alg_hcoper, batchblas, batchhvec, batchsize, blksize*2, blksize0);
            if(rank==0 && verbose>1){
               std::cout << " rank=" << rank
                  << " Hxlst.size=" << Hxlst.size()
                  << " Hmmtasks.totsize=" << Hmmtask.totsize
                  << " batchsize=" << Hmmtask.batchsize 
                  << " nbatch=" << Hmmtask.nbatch 
                  << std::endl;
               if(verbose>2){
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
      }

} // ctns

#endif
