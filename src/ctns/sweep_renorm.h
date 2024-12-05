#ifndef SWEEP_RENORM_H
#define SWEEP_RENORM_H

#include "oper_functors.h"
#include "oper_renorm_kernel.h"
#include "symbolic_formulae_renorm.h"
#include "sadmrg/symbolic_formulae_renorm_su2.h"
#include "symbolic_kernel_renorm.h"
#include "symbolic_kernel_renorm2.h"
#include "preprocess_rformulae.h"
#include "preprocess_renorm.h"
#include "preprocess_renorm_batch.h"
#ifdef GPU
#include "preprocess_renorm_batchGPU.h"
#endif

namespace ctns{

   template <typename Qm, typename Tm, typename QTm>
      struct Renorm_wrapper{
         public:
            void kernel(const std::string superblock,
                  const bool is_same,
                  const bool skipId,
                  const bool ifmps,
                  const QTm& site,
                  const QTm& site2,
                  const qoper_dict<Qm::ifabelian,Tm>& qops1,
                  const qoper_dict<Qm::ifabelian,Tm>& qops2,
                  qoper_dict<Qm::ifabelian,Tm>& qops,
                  const integral::two_body<Tm>& int2e,
                  const input::schedule& schd,
                  const int& size,
                  const int& rank,
                  const int& maxthreads,
                  dot_timing& timing,
                  const std::string fname,
                  const std::string fmmtask);
            // GEMM and GEMV list
            void init_Rlist(const bool ifSingle,
                  const bool ifDirect,
                  const std::string superblock,
                  const bool skipId,
                  const QTm& site,
                  const QTm& site2,
                  const qoper_dict<Qm::ifabelian,Tm>& qops,
                  const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
                  const bool debug);
            // generate Hmmtasks
            void init_Rmmtask(const bool ifSingle,
                  const int& batchblas,
                  const std::tuple<int,int,int>& batchrenorm,
                  const std::string& fmmtask,
                  const int rank,
                  const int verbose);
            // cleanup
            void finalize();
         private:
            int alg_renorm;
            int alg_rinter;
            int alg_rcoper;
            std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
            Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
            renorm_tasks<Tm> rtasks;
            size_t worktot=0;
            rintermediates<Qm::ifabelian,Tm> rinter;
            Rlist<Tm> Rlst; 
            Rlist2<Tm> Rlst2; 
            RMMtask<Tm> Rmmtask;
            RMMtasks<Tm> Rmmtasks;
            size_t blksize=0, blksize0=0;
            double cost=0.0;
            Tm* workspace = nullptr;
#ifdef GPU
            Tm* dev_site = nullptr;
            Tm* dev_site2 = nullptr;
            Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
            Tm* dev_workspace = nullptr;
            Tm* dev_red = nullptr;
#endif
            size_t maxbatch=0, batchsize=0, gpumem_site=0, gpumem_batch=0;
      };

   template <typename Qm, typename Tm, typename QTm>
      void Renorm_wrapper<Qm,Tm,QTm>::kernel(const std::string superblock,
            const bool is_same,
            const bool skipId,
            const bool ifmps,
            const QTm& site,
            const QTm& site2,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const int& size,
            const int& rank,
            const int& maxthreads,
            dot_timing& timing,
            const std::string fname,
            const std::string fmmtask){
         // basic setup
         const bool ifab = Qm::ifabelian;
         alg_renorm = schd.ctns.alg_renorm;
         alg_rinter = schd.ctns.alg_rinter;
         alg_rcoper = schd.ctns.alg_rcoper;
         // consistency check
         if(schd.ctns.ifdistc && !ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }
         if(!ifab && alg_renorm < 4){
            std::cout << "error: use alg_renorm >= 4 for non-Abelian case! alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         }
         if(ifab && Qm::ifkr && alg_renorm >=4){
            std::cout << "error: alg_renorm >= 4 does not support complex yet! GEMM with conj is needed." << std::endl;
            exit(1); 
         }
         if(alg_renorm < 10 and alg_rinter == 2){
            std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter<2" << std::endl;
            exit(1);
         }
         if(alg_renorm > 10 and alg_rinter != 2){
            std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter=2" << std::endl;
            exit(1);
         }
         // intermediates
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const qoper_dictmap<ifab,Tm> qops_dict = {{block1,qops1}, {block2,qops2}};
         if(superblock == "lc"){
            opaddr[0] = qops1._data;
            opaddr[2] = qops2._data;
         }else if(superblock == "cr"){
            opaddr[2] = qops1._data;
            opaddr[1] = qops2._data;
         }else if(superblock == "lr"){
            opaddr[0] = qops1._data;
            opaddr[1] = qops2._data;
         }
#ifdef GPU
         if(superblock == "lc"){
            dev_opaddr[0] = qops1._dev_data;
            dev_opaddr[2] = qops2._dev_data;
         }else if(superblock == "cr"){
            dev_opaddr[2] = qops1._dev_data;
            dev_opaddr[1] = qops2._dev_data;
         }else if(superblock == "lr"){
            dev_opaddr[0] = qops1._dev_data;
            dev_opaddr[1] = qops2._dev_data;
         }
#endif

         // setup formulae
         timing.tf1 = tools::get_time();
         if(alg_renorm > 0){
            rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, schd.ctns.sort_formulae, schd.ctns.ifdist1, 
                  schd.ctns.ifdistc, schd.ctns.verbose>0);
         }
         timing.tf2 = tools::get_time();

         // perform renormalization
         if(alg_renorm == 0){

            // oldest version
            auto rfuns = oper_renorm_functors(superblock, site2, int2e, qops1, qops2, qops, schd.ctns.ifdist1);
            // initialize of qops
            oper_renorm_kernel(superblock, rfuns, site, qops, schd.ctns.verbose);

         }else if(alg_renorm == 1){

            // symbolic formulae + dynamic allocation of memory
            // initialization of qops inside
            symbolic_kernel_renorm(superblock, rtasks, site, site2, qops1, qops2, qops, skipId, schd.ctns.ifdist1, schd.ctns.verbose);

         }else if(alg_renorm == 2){

            // symbolic formulae + preallocation of workspace
            if(!is_same){
               std::cout << "error: alg_renorm=2 only work for icomb=icomb2!" << std::endl;
               exit(1);
            }
            // initialization of qops inside 
            symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, skipId, schd.ctns.ifdist1, schd.ctns.verbose);

         }else if(alg_renorm == 4){

            // CPU: symbolic formulae + rintermediates + preallocation of workspace
            timing.tf3 = tools::get_time();
            timing.tf4 = tools::get_time();
            const bool ifDirect = false;
            const int batchgemv = 1;
            rinter.init(ifDirect, alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, rank==0);
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist(ifDirect, alg_rcoper, superblock, 
                  qops, qops_dict, oploc, opaddr, rtasks, site, site2, skipId, rinter,
                  Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            get_MMlist2(Rlst);
            timing.tf6 = tools::get_time();
            timing.tf7 = tools::get_time();
            timing.tf8 = tools::get_time();

            worktot = maxthreads*(blksize*2+qops._size);
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            // initialization of qops inside
            preprocess_renorm(qops._data, site._data, site2._data, size, rank, qops._size, blksize, Rlst, opaddr);
            timing.tf9 = tools::get_time();

         }else if(alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            timing.tf3 = tools::get_time();
            timing.tf4 = tools::get_time();
            const bool ifSingle = alg_renorm > 7;
            const bool ifDirect = alg_renorm % 2 == 1;
            const int batchgemv = 1;
            rinter.init(ifDirect, alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, rank==0);
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            this->init_Rlist(ifSingle, ifDirect, superblock, skipId, site, site2, qops, qops_dict, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            if(blksize > 0){
               // determine batchsize dynamically
               size_t blocksize = 2*blksize+blksize0;
               preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch,
                     batchsize, worktot);
               if(rank==0 && schd.ctns.verbose>0){
                  std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                     << " blksize0=" << blksize0 << " batchsize=" << batchsize
                     << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                     << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
               }
               workspace = new Tm[worktot];

               // generate Rmmtasks
               const int batchblas = schd.ctns.alg_rinter; // use the same keyword for GEMM_batch
               auto batchrenorm = std::make_tuple(batchblas,batchblas,batchblas); 
               this->init_Rmmtask(ifSingle, batchblas, batchrenorm, fmmtask, rank, schd.ctns.verbose);
            } // blksize>0
            timing.tf7 = tools::get_time();

            // initialization of qops
            memset(qops._data, 0, qops._size*sizeof(Tm));
            timing.tf8 = tools::get_time();

            // kernel
            if(blksize > 0){
               if(!ifSingle){
                  if(!ifDirect){ 
                     preprocess_renorm_batch(qops._data, site._data, site2._data, size, rank, qops._size,
                           Rmmtasks, opaddr, workspace);
                  }else{
                     opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                     preprocess_renorm_batchDirect(qops._data, site._data, site2._data, size, rank, qops._size,
                           Rmmtasks, opaddr, workspace, 
                           rinter._data);
                  }
               }else{
                  if(!ifDirect){ 
                     preprocess_renorm_batchSingle(qops._data, site._data, site2._data, size, rank, qops._size,
                           Rmmtask, opaddr, workspace);
                  }else{
                     opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                     preprocess_renorm_batchDirectSingle(qops._data, site._data, site2._data, size, rank, qops._size,
                           Rmmtask, opaddr, workspace, 
                           rinter._data);
                  }
               }
            } // blksize>0
            timing.tf9 = tools::get_time();

#ifdef GPU
         }else if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace

            // initialization of qops
            qops.allocate_gpu(true);
            size_t opertot = qops1.size() + qops2.size() + qops.size();
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf3 = tools::get_time();

            // bra
            gpumem_site = sizeof(Tm)*site.size();
            dev_site = (Tm*)GPUmem.allocate(gpumem_site);
            GPUmem.to_gpu(dev_site, site._data, gpumem_site);
            // ket
            if(is_same){
               dev_site2 = dev_site;
            }else{
               size_t gpumem_site2 = sizeof(Tm)*site2.size();
               dev_site2 = (Tm*)GPUmem.allocate(gpumem_site2);
               GPUmem.to_gpu(dev_site2, site2._data, gpumem_site2);
               gpumem_site += gpumem_site2;
            }
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,site)(GB)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_site/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf4 = tools::get_time();

            // compute hintermediates on CPU
            const bool ifSingle = alg_renorm > 17;
            const bool ifDirect = alg_renorm % 2 == 1;
            const int batchgemv = std::get<0>(schd.ctns.batchrenorm);
            rinter.init(ifDirect, schd.ctns.alg_rinter, batchgemv, qops_dict, oploc, dev_opaddr, rtasks, rank==0);
            size_t gpumem_rinter = sizeof(Tm)*rinter.size();
            if(rank==0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,site,rinter)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_site/std::pow(1024.0,3) 
                  << "," << gpumem_rinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            this->init_Rlist(ifSingle, ifDirect, superblock, skipId, site, site2, qops, qops_dict, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            if(blksize > 0){
               // Determine batchsize dynamically & GPUmem.allocate
               size_t blocksize = 2*blksize+blksize0+1;
               preprocess_gpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, 0, rank, 
                     batchsize, gpumem_batch);
               if(rank==0 && schd.ctns.verbose>0){
                  size_t used = GPUmem.used();
                  size_t avail = GPUmem.available(rank);
                  size_t total = used + avail;
                  std::cout << "rank=" << rank
                     << " GPUmem(GB): used=" << used/std::pow(1024.0,3)
                     << " avail=" << avail/std::pow(1024.0,3)
                     << " total=" << total/std::pow(1024.0,3)
                     << " batch[need]=" << gpumem_batch/std::pow(1024.0,3)
                     << " blksize=" << blksize
                     << " blksize0=" << blksize0 
                     << " batchsize=" << batchsize 
                     << std::endl;
               }
               dev_workspace = (Tm*)GPUmem.allocate(gpumem_batch);
               if(rank==0 && schd.ctns.verbose>0){
                  std::cout << "rank=" << rank
                     << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                     << " (oper,site,rinter,batch)=" << gpumem_oper/std::pow(1024.0,3) 
                     << "," << gpumem_site/std::pow(1024.0,3) 
                     << "," << gpumem_rinter/std::pow(1024.0,3) 
                     << "," << gpumem_batch/std::pow(1024.0,3)
                     << std::endl;
               }
 
               // generate Rmmtasks given batchsize
               const int batchblas = 2; // GPU
               this->init_Rmmtask(ifSingle, batchblas, schd.ctns.batchrenorm, fmmtask, rank, schd.ctns.verbose);
            } // blksize>0
            timing.tf7 = tools::get_time();
            timing.tf8 = tools::get_time();

            // kernel
            if(blksize > 0){
               dev_red = dev_workspace + batchsize*(blksize*2+blksize0);
               if(!ifSingle){
                  if(!ifDirect){
                     preprocess_renorm_batchGPU(qops._dev_data, dev_site, dev_site2, 
                           size, rank, qops._size,
                           Rmmtasks, dev_opaddr, dev_workspace, 
                           dev_red);
                  }else{
                     dev_opaddr[4] = dev_workspace + batchsize*blksize*2; // tmpspace for intermediates
                     preprocess_renorm_batchDirectGPU(qops._dev_data, dev_site, dev_site2, 
                           size, rank, qops._size,
                           Rmmtasks, dev_opaddr, dev_workspace, 
                           rinter._dev_data, dev_red);
                  }
               }else{
                  if(!ifDirect){
                     preprocess_renorm_batchGPUSingle(qops._dev_data, dev_site, dev_site2, 
                           size, rank, qops._size,
                           Rmmtask, dev_opaddr, dev_workspace, 
                           dev_red);
                  }else{
                     dev_opaddr[4] = dev_workspace + batchsize*blksize*2; // tmpspace for intermediates
                     preprocess_renorm_batchDirectGPUSingle(qops._dev_data, dev_site, dev_site2, 
                           size, rank, qops._size,
                           Rmmtask, dev_opaddr, dev_workspace, 
                           rinter._dev_data, dev_red);
                  }
               }
            } // blksize>0
            timing.tf9 = tools::get_time();
#endif // GPU
       
         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm
      }

   // finalize
   template <typename Qm, typename Tm, typename QTm>
      void Renorm_wrapper<Qm,Tm,QTm>::finalize(){
         if(alg_renorm==6 || alg_renorm==7 ||
               alg_renorm==8 || alg_renorm==9){
            delete[] workspace;
         }
#ifdef GPU
         if(alg_renorm>10){
            GPUmem.deallocate(dev_site, gpumem_site);
            if(blksize > 0) GPUmem.deallocate(dev_workspace, gpumem_batch);
         }
#endif
      }

   // GEMM and GEMV list
   template <typename Qm, typename Tm, typename QTm>
      void Renorm_wrapper<Qm,Tm,QTm>::init_Rlist(const bool ifSingle,
            const bool ifDirect,
            const std::string superblock,
            const bool skipId,
            const QTm& site,
            const QTm& site2,
            const qoper_dict<Qm::ifabelian,Tm>& qops,
            const qoper_dictmap<Qm::ifabelian,Tm>& qops_dict,
            const bool debug){
         if(!ifSingle){
            preprocess_formulae_Rlist2(ifDirect, alg_rcoper, superblock, 
                  qops, qops_dict, oploc, opaddr, rtasks, site, site2, skipId, rinter,
                  Rlst2, blksize, blksize0, cost, debug);
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
         }else{
            preprocess_formulae_Rlist(ifDirect, alg_rcoper, superblock, 
                  qops, qops_dict, oploc, opaddr, rtasks, site, site2, skipId, rinter,
                  Rlst, blksize, blksize0, cost, debug);
            maxbatch = Rlst.size();
         }
         if(!ifDirect) assert(blksize0 == 0);
      } 

   // generate Hmmtasks
   template <typename Qm, typename Tm, typename QTm>
      void Renorm_wrapper<Qm,Tm,QTm>::init_Rmmtask(const bool ifSingle,
            const int& batchblas,
            const std::tuple<int,int,int>& batchrenorm,
            const std::string& fmmtask,
            const int rank,
            const int verbose){
         if(!ifSingle){
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], batchblas, batchrenorm, batchsize, blksize*2, blksize0);
               if(rank==0 && verbose>1 && Rlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i
                     << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                     << " batchsize=" << Rmmtasks[i].batchsize
                     << " nbatch=" << Rmmtasks[i].nbatch
                     << std::endl;
               }
            } // i
            if(fmmtask.size()>0) save_mmtask(Rmmtasks, fmmtask);
         }else{
            Rmmtask.init(Rlst, batchblas, batchrenorm, batchsize, blksize*2, blksize0);
            if(rank==0 && verbose>1){
               std::cout << " rank=" << rank 
                  << " Rlst.size=" << Rlst.size()
                  << " Rmmtask.totsize=" << Rmmtask.totsize
                  << " batchsize=" << Rmmtask.batchsize 
                  << " nbatch=" << Rmmtask.nbatch 
                  << std::endl;
               if(verbose>2){
                  for(int k=0; k<Rmmtask.nbatch; k++){
                     for(int i=0; i<Rmmtask.mmbatch2[k].size(); i++){
                        if(Rmmtask.mmbatch2[k][i].size==0) continue;
                        std::cout << " Rmmbatch2: k/nbatch=" << k << "/" << Rmmtask.nbatch
                           << " i=" << i << " size=" << Rmmtask.mmbatch2[k][i].size
                           << " group=" << Rmmtask.mmbatch2[k][i].gsta.size()-1
                           << " average=" << Rmmtask.mmbatch2[k][i].size/(Rmmtask.mmbatch2[k][i].gsta.size()-1)
                           << std::endl;
                     }
                  }
               }
            }
            if(fmmtask.size()>0) save_mmtask(Rmmtask, fmmtask);
         }
      }

} // ctns

#endif
