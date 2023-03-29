#ifndef OPER_RENORM_H
#define OPER_RENORM_H

#include "sweep_data.h"
#include "oper_timer.h"
#include "oper_functors.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"
#include "oper_rbasis.h"
#include "oper_renorm_kernel.h"
#include "symbolic_kernel_renorm.h"
#include "symbolic_kernel_renorm2.h"
#include "preprocess_rformulae.h"
#include "preprocess_renorm.h"
#include "preprocess_renorm_batch.h"
#include "preprocess_renorm_batch2.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "preprocess_renorm_batchGPU.h"
#include "preprocess_renorm_batch2GPU.h"
#endif

namespace ctns{

   const bool debug_oper_renorm = false;
   extern const bool debug_oper_renorm;

   const bool debug_oper_rbasis = false;
   extern const bool debug_oper_rbasis;

   const double thresh_opdiff = 1.e-10;
   extern const double thresh_opdiff;

   // renormalize operators
   template <typename Km, typename Tm>
      size_t oper_renorm_opAll(const std::string superblock,
            const comb<Km>& icomb,
            const comb_coord& p,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            oper_dict<Tm>& qops,
            const std::string fname,
            dot_timing& timing){
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const int alg_renorm = schd.ctns.alg_renorm;
         const int isym = Km::isym;
         const bool ifkr = Km::ifkr;
         const bool sort_formulae = schd.ctns.sort_formulae;
         const bool ifdist1 = schd.ctns.ifdist1;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::oper_renorm_opAll coord=" << p 
               << " superblock=" << superblock 
               << " isym=" << isym 
               << " ifkr=" << ifkr
               << " alg_renorm=" << alg_renorm	
               << " mpisize=" << size
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         timing.tf0 = tools::get_time(); 

         // 0. setup basic information for qops
         qops.isym = isym;
         qops.ifkr = ifkr;
         qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
         // rest of spatial orbital indices
         const auto& node = icomb.topo.get_node(p);
         const auto& rindex = icomb.topo.rindex;
         const auto& site = icomb.sites[rindex.at(p)];
         if(superblock == "cr"){
            qops.krest = node.lorbs;
            qops.qbra = site.info.qrow;
            qops.qket = site.info.qrow; 
         }else if(superblock == "lc"){
            qops.krest = node.rorbs;
            qops.qbra = site.info.qcol;
            qops.qket = site.info.qcol;
         }else if(superblock == "lr"){
            qops.krest = node.corbs;
            qops.qbra = site.info.qmid;
            qops.qket = site.info.qmid;
         }
         qops.oplist = "CABPQSH";
         qops.mpisize = size;
         qops.mpirank = rank;
         qops.ifdist2 = true;
         // initialize memory 
         qops.allocate_memory();

         //-------------------------------
         // 1. kernel for renormalization
         //-------------------------------
         oper_timer.start();
         const bool debug_formulae = schd.ctns.verbose>0;
         size_t worktot=0;
         // intermediates      
         rintermediates<Tm> rinter;
         Rlist<Tm> Rlst;
         Rlist2<Tm> Rlst2;
         RMMtasks<Tm> Rmmtasks;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr}; // {l,r,c1,c2,i}
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const oper_dictmap<Tm> qops_dict = {{block1,qops1}, {block2,qops2}};
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
         size_t blksize=0, blksize0=0;
         double cost=0.0;
         Tm* workspace;
#ifdef GPU
         Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
         Tm* dev_oper = nullptr;
         Tm* dev_workspace = nullptr;
         Tm* dev_site = nullptr;
         Tm* dev_qops = nullptr;
         size_t GPUmem_used = 0;
#endif
         if(Km::ifkr && alg_renorm >=4){
            std::cout << "error: alg_renorm >= 4 does not support complex yet!" << std::endl;
            exit(1); 
         }
         timing.tf1 = tools::get_time();
         if(alg_renorm == 0){

            // oldest version
            auto rfuns = oper_renorm_functors(superblock, site, int2e, qops1, qops2, qops, ifdist1);
            oper_renorm_kernel(superblock, rfuns, site, qops, schd.ctns.verbose);

         }else if(alg_renorm == 1){

            // symbolic formulae + dynamic allocation of memory
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, 
                  debug_formulae);
            symbolic_kernel_renorm(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);

         }else if(alg_renorm == 2){

            // symbolic formulae + preallocation of workspace
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            worktot = symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);
         
         }else if(alg_renorm == 4){

            // CPU: symbolic formulae + rintermediates + preallocation of workspace
            timing.tf2 = tools::get_time();
            
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            timing.tf3 = tools::get_time();
            
            // generation of renormalization block [lc/lr/cr]
            rinter.init(schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst, blksize, cost, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            get_MMlist(Rlst, schd.ctns.hxorder);

            worktot = maxthreads*(blksize*2+qops._size);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            timing.tf7 = tools::get_time();

            preprocess_renorm(qops._data, site._data, size, rank, qops._size, blksize, Rlst, opaddr);
            timing.tf8 = tools::get_time();

         }else if(alg_renorm == 6){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            if(schd.ctns.alg_rinter != 0 and schd.ctns.alg_rinter != 1){
               std::cout << "error: alg_renorm=6 should be used with alg_rinter=0 or 1!" << std::endl;
               exit(1);
            }
            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }
            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            timing.tf3 = tools::get_time();

            // generation of renormalization block [lc/lr/cr]
            rinter.init(schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist2(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
            size_t batchsize = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;

            // generate Rmmtasks
            const int batchblas = 1;
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2);
               if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i
                      << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                      << " batchsize=" << Rmmtasks[i].batchsize
                      << " nbatch=" << Rmmtasks[i].nbatch
                      << std::endl;
               }
            }

            worktot = batchsize*blksize*2;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            timing.tf7 = tools::get_time();
            
            workspace = new Tm[worktot];
            preprocess_renorm_batch(qops._data, site._data, size, rank, qops._size,
                                    Rmmtasks, opaddr, workspace);
            timing.tf8 = tools::get_time();

         }else if(alg_renorm == 7){

            // BatchCPU: symbolic formulae + rintermediates [on the fly] + preallocation of workspace
            if(schd.ctns.alg_rinter != 1){
               std::cout << "error: alg_renorm=7 should be used with alg_rinter=1!" << std::endl;
               exit(1);
            }
            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }
            timing.tf2 = tools::get_time();
            
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            timing.tf3 = tools::get_time();
            
            // generation of renormalization block [lc/lr/cr]
            rinter.initDirect(schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist2Direct(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
            size_t batchsize = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;

            // generate Rmmtasks
            const int batchblas = 1;
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i
                      << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                      << " batchsize=" << Rmmtasks[i].batchsize
                      << " nbatch=" << Rmmtasks[i].nbatch
                      << std::endl;
               }
            }

            worktot = batchsize*(blksize*2+blksize0);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize
                  << " blksize0=" << blksize0 << " batchsize=" << batchsize
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            timing.tf7 = tools::get_time();
            
            workspace = new Tm[worktot];
            opaddr[4] = workspace + batchsize*(blksize*2); // memory layout [workspace|inter]
            preprocess_renorm_batch2(qops._data, site._data, size, rank, qops._size,
                                     Rmmtasks, opaddr, workspace, rinter._data);
            timing.tf8 = tools::get_time();

#ifdef GPU
         }else if(alg_renorm == 16){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << std::endl;
            }
            if(GPUmem.used() != 0){ 
               std::cout << "error: there should not be any use of GPU memory at this point!" << std::endl;
               std::cout << "GPUmem.used=" << GPUmem.used() << std::endl;
               exit(1);
            }

            // allocate memery on GPU & copy qops
            size_t opertot = qops1.size() + qops2.size();
            size_t GPUmem_oper = sizeof(Tm)*opertot;
            dev_oper = (Tm*)GPUmem.allocate(GPUmem_oper);
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            dev_opaddr[0] = dev_oper;
            if(superblock == "lc"){
               dev_opaddr[1] = dev_opaddr[0] + qops1.size();
               dev_opaddr[2] = dev_opaddr[1];
               dev_opaddr[3] = dev_opaddr[2] + qops2.size();
            }else if(superblock == "cr"){
               dev_opaddr[1] = dev_opaddr[0];
               dev_opaddr[2] = dev_opaddr[1] + qops2.size();
               dev_opaddr[3] = dev_opaddr[2] + qops1.size();
            }else if(superblock == "lr"){
               dev_opaddr[1] = dev_opaddr[0] + qops1.size();
               dev_opaddr[2] = dev_opaddr[1] + qops2.size();
               dev_opaddr[3] = dev_opaddr[2];
            }
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_opaddr[oploc[block1]], qops1._data, qops1.size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[oploc[block2]], qops2._data, qops2.size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_opaddr[oploc[block1]], qops1._data, qops1.size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[oploc[block2]], qops2._data, qops2.size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            size_t GPUmem_qops = sizeof(Tm)*qops.size();
            size_t GPUmem_site = sizeof(Tm)*site.size();
            dev_qops = (Tm*)GPUmem.allocate(GPUmem_qops);
            dev_site = (Tm*)GPUmem.allocate(GPUmem_site);
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_qops/std::pow(1024.0,3) 
                  << "," << GPUmem_site/std::pow(1024.0,3) 
                  << std::endl;
            }
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_site, site._data, site.size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_site, site._data, site.size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);

            timing.tf3 = tools::get_time();

            //------------------------------
            // intermediates
            //------------------------------
            if(schd.ctns.alg_rinter != 2){
               rinter.init(schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            }else{
               rinter.init(schd.ctns.alg_rinter, qops_dict, oploc, dev_opaddr, rtasks, debug);
            }
            timing.tf4 = tools::get_time();

            size_t GPUmem_rinter = sizeof(Tm)*rinter.size();
            // copy from CPU to GPU 
            if(schd.ctns.alg_rinter != 2){
               dev_opaddr[4] = (Tm*)GPUmem.allocate(GPUmem_rinter);
#ifdef USE_HIP
               HIP_CHECK(hipMemcpy(dev_opaddr[4], rinter._value.data(), GPUmem_rinter, hipMemcpyHostToDevice));
#else
               CUDA_CHECK(cudaMemcpy(dev_opaddr[4], rinter._value.data(), GPUmem_rinter, cudaMemcpyHostToDevice));
#endif// USE_HIP
            }
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site,rinter)(GB)=" << GPUmem_oper/std::pow(1024.0,3)
                  << "," << GPUmem_qops/std::pow(1024.0,3) 
                  << "," << GPUmem_site/std::pow(1024.0,3) 
                  << "," << GPUmem_rinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist2(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);

            timing.tf6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
            maxbatch = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;
            
            // Determine batchsize dynamically: following sweep_twodot.h [additional N for reduction]
            size_t batchsize = 0;
            size_t GPUmem_reserved = GPUmem_oper + GPUmem_rinter + GPUmem_qops + GPUmem_site + 48;
            if(GPUmem.size() > GPUmem_reserved){
               batchsize = std::floor(double(GPUmem.size() - GPUmem_reserved)/(sizeof(Tm)*blksize*2 + 113));
               batchsize = (maxbatch < batchsize)? maxbatch : batchsize; // sufficient
               if(batchsize == 0 && maxbatch != 0){
                  std::cout << "error: in sufficient GPU memory: batchsize=0!" << std::endl;
                  exit(1);
               }
            }else{
               std::cout << "error: in sufficient GPU memory for batchGEMM! already reserved:" << std::endl;
               std::cout << "GPUmem.size=" << GPUmem.size() << " GPUmem.used=" << GPUmem.used()
                  << " GPUmem_reserved=" << GPUmem_reserved << " (oper,rinter,qops,site)=" 
                  << GPUmem_oper << "," << GPUmem_rinter << "," << GPUmem_qops << "," << GPUmem_site 
                  << std::endl;
               exit(1);
            }
            size_t GPUmem_batch = sizeof(Tm)*batchsize*(blksize*2+1);
            dev_workspace = (Tm*)GPUmem.allocate(GPUmem_batch);
            GPUmem_used = GPUmem.used(); // later used in deallocate
            if(schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site,rinter,batch)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_qops/std::pow(1024.0,3)
                  << "," << GPUmem_site/std::pow(1024.0,3)
                  << "," << GPUmem_rinter/std::pow(1024.0,3)
                  << "," << GPUmem_batch/std::pow(1024.0,3)
                  << " batchsize=" << batchsize
                  << std::endl;
            }

            // generate Rmmtasks
            const int batchblas = 2;
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2);
               if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i
                      << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                      << " batchsize=" << Rmmtasks[i].batchsize
                      << " nbatch=" << Rmmtasks[i].nbatch
                      << std::endl;
               }
            }
            timing.tf7 = tools::get_time();

            // kernel
            Tm* dev_red = dev_workspace + batchsize*blksize*2;
            preprocess_renorm_batchGPU(dev_qops, dev_site, size, rank, qops._size,
                                    Rmmtasks, dev_opaddr, dev_workspace, dev_red);
            
            timing.tf8 = tools::get_time();

            // copy results back to CPU
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(qops._data, dev_qops, qops.size()*sizeof(Tm), hipMemcpyDeviceToHost));
#else
            CUDA_CHECK(cudaMemcpy(qops._data, dev_qops, qops.size()*sizeof(Tm), cudaMemcpyDeviceToHost));
#endif

         }else if(alg_renorm == 17){

            // BatchCPU: symbolic formulae + rintermediates [on the fly] + preallocation of workspace
            if(schd.ctns.alg_rinter != 2){
               std::cout << "error: alg_renorm=17 should be used with alg_rinter=2!" << std::endl;
               exit(1);
            }
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << std::endl;
            }
            if(GPUmem.used() != 0){ 
               std::cout << "error: there should not be any use of GPU memory at this point!" << std::endl;
               std::cout << "GPUmem.used=" << GPUmem.used() << std::endl;
               exit(1);
            }
          
            // allocate memery on GPU & copy qops
            size_t opertot = qops1.size() + qops2.size();
            size_t GPUmem_oper = sizeof(Tm)*opertot;
            dev_oper = (Tm*)GPUmem.allocate(GPUmem_oper);
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            dev_opaddr[0] = dev_oper;
            if(superblock == "lc"){
               dev_opaddr[1] = dev_opaddr[0] + qops1.size();
               dev_opaddr[2] = dev_opaddr[1];
               dev_opaddr[3] = dev_opaddr[2] + qops2.size();
            }else if(superblock == "cr"){
               dev_opaddr[1] = dev_opaddr[0];
               dev_opaddr[2] = dev_opaddr[1] + qops2.size();
               dev_opaddr[3] = dev_opaddr[2] + qops1.size();
            }else if(superblock == "lr"){
               dev_opaddr[1] = dev_opaddr[0] + qops1.size();
               dev_opaddr[2] = dev_opaddr[1] + qops2.size();
               dev_opaddr[3] = dev_opaddr[2];
            }
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_opaddr[oploc[block1]], qops1._data, qops1.size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[oploc[block2]], qops2._data, qops2.size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_opaddr[oploc[block1]], qops1._data, qops1.size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[oploc[block2]], qops2._data, qops2.size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            size_t GPUmem_qops = sizeof(Tm)*qops.size();
            size_t GPUmem_site = sizeof(Tm)*site.size();
            dev_qops = (Tm*)GPUmem.allocate(GPUmem_qops);
            dev_site = (Tm*)GPUmem.allocate(GPUmem_site);
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_qops/std::pow(1024.0,3) 
                  << "," << GPUmem_site/std::pow(1024.0,3) 
                  << std::endl;
            }
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_site, site._data, site.size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_site, site._data, site.size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            timing.tf2 = tools::get_time();
            
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
 
            timing.tf3 = tools::get_time();

            //------------------------------
            // intermediates
            //------------------------------
            rinter.initDirect(schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();

            size_t GPUmem_rinter = sizeof(Tm)*rinter.size();
            // copy from CPU to GPU 
            if(schd.ctns.alg_rinter != 2){
               dev_opaddr[4] = (Tm*)GPUmem.allocate(GPUmem_rinter);
#ifdef USE_HIP
               HIP_CHECK(hipMemcpy(dev_opaddr[4], rinter._value.data(), GPUmem_rinter, hipMemcpyHostToDevice));
#else
               CUDA_CHECK(cudaMemcpy(dev_opaddr[4], rinter._value.data(), GPUmem_rinter, cudaMemcpyHostToDevice));
#endif// USE_HIP
            }
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site,rinter)(GB)=" << GPUmem_oper/std::pow(1024.0,3)
                  << "," << GPUmem_qops/std::pow(1024.0,3) 
                  << "," << GPUmem_site/std::pow(1024.0,3) 
                  << "," << GPUmem_rinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist2Direct(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            
            timing.tf6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
            maxbatch = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;

            // Determine batchsize dynamically: following sweep_twodot.h [additional N for reduction]
            size_t batchsize = 0;
            size_t GPUmem_reserved = GPUmem_oper + GPUmem_rinter + GPUmem_qops + GPUmem_site + 88;
            if(GPUmem.size() > GPUmem_reserved){
               batchsize = std::floor(double(GPUmem.size() - GPUmem_reserved)/(sizeof(Tm)*(blksize*2+blksize0) + 201));
               batchsize = (maxbatch < batchsize)? maxbatch : batchsize; // sufficient
               if(batchsize == 0 && maxbatch != 0){
                  std::cout << "error: in sufficient GPU memory: batchsize=0!" << std::endl;
                  exit(1);
               }
            }else{
               std::cout << "error: in sufficient GPU memory for batchGEMM! already reserved:" << std::endl;
               std::cout << "GPUmem.size=" << GPUmem.size() << " GPUmem.used=" << GPUmem.used()
                  << " GPUmem_reserved=" << GPUmem_reserved << " (oper,rinter,qops,site)=" 
                  << GPUmem_oper << "," << GPUmem_rinter << "," << GPUmem_qops << "," << GPUmem_site 
                  << std::endl;
               exit(1);
            }
            size_t GPUmem_batch = sizeof(Tm)*batchsize*(blksize*2+blksize0+1);
            dev_workspace = (Tm*)GPUmem.allocate(GPUmem_batch);
            GPUmem_used = GPUmem.used(); // later used in deallocate
            if(schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,qops,site,rinter,batch)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_qops/std::pow(1024.0,3)
                  << "," << GPUmem_site/std::pow(1024.0,3)
                  << "," << GPUmem_rinter/std::pow(1024.0,3)
                  << "," << GPUmem_batch/std::pow(1024.0,3)
                  << " batchsize=" << batchsize
                  << std::endl;
            }

            // generate Rmmtasks
            const int batchblas = 2;
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i
                      << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                      << " batchsize=" << Rmmtasks[i].batchsize
                      << " nbatch=" << Rmmtasks[i].nbatch
                      << std::endl;
               }
            }
            timing.tf7 = tools::get_time();

            dev_opaddr[4] = dev_workspace + batchsize*blksize*2; // tmpspace for intermediates
            Tm* dev_red = dev_opaddr[4] + batchsize*blksize0; // tmpspace for coefficients in reduction
           
            // debug 
            Tm* tmp = new Tm[qops.size()];
            CUDA_CHECK(cudaMemcpy(tmp, dev_qops, qops.size()*sizeof(Tm), cudaMemcpyDeviceToHost));
            double error_init = linalg::xnrm2(qops.size(), tmp);
            delete[] tmp;
            std::cout << "rank=" << rank << " error_init=" << error_init << std::endl;
            
            preprocess_renorm_batch2GPU(dev_qops, dev_site, size, rank, qops._size,
                                     Rmmtasks, dev_opaddr, dev_workspace, rinter._data, dev_red);

            timing.tf8 = tools::get_time();

            // copy results back to CPU
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(qops._data, dev_qops, qops.size()*sizeof(Tm), hipMemcpyDeviceToHost));
#else
            CUDA_CHECK(cudaMemcpy(qops._data, dev_qops, qops.size()*sizeof(Tm), cudaMemcpyDeviceToHost));
#endif

#endif // GPU

         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm
         timing.tf9 = tools::get_time();

         // debug 
         if(debug_oper_renorm){
            const int target = -1;
            std::cout << "\nlzd:qops:" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cnew",2);
               }
            }
            Tm* data0 = new Tm[qops._size];
            linalg::xcopy(qops._size, qops._data, data0);
            
            // alg_renorm=2: symbolic formulae + preallocation of workspace
            memset(qops._data, 0, qops._size*sizeof(Tm));
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            worktot = symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);
            std::cout << "\nlzd:qops: ref" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second[ref]=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cref",2);
               }
            }
            Tm* data1 = new Tm[qops._size];
            linalg::xcopy(qops._size, qops._data, data1);

            linalg::xaxpy(qops._size, -1.0, data0, qops._data);
            auto diff = linalg::xnrm2(qops._size, qops._data);
            std::cout << "\nlzd:qops-diff" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second[diff]=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cdiff",2);
               }
            }
            std::cout << "total diff=" << diff << std::endl;
            linalg::xcopy(qops._size, data0, qops._data);
            delete[] data0;
            delete[] data1;
            if(diff > thresh_opdiff) exit(1);
         }

         // free tmp space on CPU
         if(alg_renorm==6 || alg_renorm==7){
            delete[] workspace;
         }
#ifdef GPU
         if(alg_renorm>10){
            GPUmem.deallocate(dev_oper, GPUmem_used);
         }
#endif

         // 2. reduce 
#ifndef SERIAL
         if(size > 1 and ifdist1){
            std::vector<Tm> top(qops._opsize);
            // Sp[iproc] += \sum_i Sp[i]
            auto opS_index = qops.oper_index_op('S');
            for(int p : opS_index){
               int iproc = distribute1(ifkr,size,p);
               auto& opS = qops('S')[p];
               int opsize = opS.size();
               mpi_wrapper::reduce(icomb.world, opS.data(), opsize, top.data(), std::plus<Tm>(), iproc);
               if(iproc == rank){ 
                  linalg::xcopy(opsize, top.data(), opS.data());
               }else{
                  opS.set_zero();
               }
            }
            // H[0] += \sum_i H[i]
            auto& opH = qops('H')[0];
            int opsize = opH.size();
            mpi_wrapper::reduce(icomb.world, opH.data(), opsize, top.data(), std::plus<Tm>(), 0);
            if(rank == 0){ 
               linalg::xcopy(opsize, top.data(), opH.data());
            }else{
               opH.set_zero();
            }
         }
#endif

         // 3. consistency check for Hamiltonian
         const auto& opH = qops('H').at(0);
         auto diffH = (opH-opH.H()).normF();
         std::cout << "check ||H-H.dagger||=" << std::scientific << std::setprecision(20) << diffH 
            << " coord=" << p << " rank=" << rank 
            << std::defaultfloat << std::setprecision(2) 
            << std::endl; 
         if(diffH > thresh_opdiff){
            std::cout <<  "error in oper_renorm: ||H-H.dagger|| is larger than thresh_opdiff=" << thresh_opdiff 
               << std::endl;
            //exit(1);
         }

         // check against explicit construction
         if(debug_oper_rbasis){
            for(const auto& key : qops.oplist){
               if(key == 'C' || key == 'A' || key == 'B'){
                  oper_check_rbasis(icomb, icomb, p, qops, key, size, rank);
               }else if(key == 'P' || key == 'Q'){
                  oper_check_rbasis(icomb, icomb, p, qops, key, int2e, int1e, size, rank);
                  // check opS and opH only if ifdist1=true   
               }else if((key == 'S' || key == 'H') and ifdist1){
                  oper_check_rbasis(icomb, icomb, p, qops, key, int2e, int1e, size, rank, ifdist1);
               }
            }
         }

         timing.tf10 = tools::get_time();
         if(debug){ 
            if(schd.ctns.verbose>0) qops.print("qops");
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
         }

            double t_tot = tools::get_duration(timing.tf10-timing.tf0); 
            double t_init = tools::get_duration(timing.tf1-timing.tf0);
            double t_cal = tools::get_duration(timing.tf9-timing.tf1);
            double t_comm = tools::get_duration(timing.tf10-timing.tf9);
            std::cout << "TIMING for Renormalization : " << t_tot 
               << " T(init/cal/comm)=" << t_init << "," << t_cal << "," << t_comm
               << " rank=" << rank
               << std::endl;
         
         return worktot;
      }

} // ctns

#endif
