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
#ifdef _OPENMP
#include <omp.h>
#endif
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "preprocess_renorm_batchGPU.h"
#endif

namespace ctns{

   const bool debug_oper_renorm = false; 
   extern const bool debug_oper_renorm;

   const bool debug_oper_rbasis = false;
   extern const bool debug_oper_rbasis;

   const double thresh_opdiff = 1.e-9;
   extern const double thresh_opdiff;

   // renormalize operators
   template <typename Km, typename Tm>
      void oper_renorm(const std::string superblock,
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
            std::cout << "ctns::oper_renorm coord=" << p 
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
         qops.allocate();
         if(debug) qops.print("qops");

         //-------------------------------
         // 1. kernel for renormalization
         //-------------------------------
         oper_timer.dot_start();
         const bool debug_formulae = schd.ctns.verbose>0;
         size_t worktot=0;
         // intermediates      
         rintermediates<Tm> rinter;
         Rlist<Tm> Rlst;
         Rlist2<Tm> Rlst2;
         RMMtask<Tm> Rmmtask;
         RMMtasks<Tm> Rmmtasks;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
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
         Tm* dev_site = nullptr;
         Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
         Tm* dev_workspace = nullptr;
         Tm* dev_red = nullptr;
#endif
         size_t batchsize, gpumem_batch;

         // consistency check
         if(Km::ifkr && alg_renorm >=4){
            std::cout << "error: alg_renorm >= 4 does not support complex yet!" << std::endl;
            exit(1); 
         }
         if(alg_renorm < 10 and schd.ctns.alg_rinter == 2){
            std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter<2" << std::endl;
            exit(1);
         }
         if(alg_renorm > 10 and schd.ctns.alg_rinter != 2){
            std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter=2" << std::endl;
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
                  size, rank, fname, sort_formulae, ifdist1, debug_formulae);
            symbolic_kernel_renorm(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);

         }else if(alg_renorm == 2){

            // symbolic formulae + preallocation of workspace
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, debug_formulae);
            symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);

         }else if(alg_renorm == 4){

            // CPU: symbolic formulae + rintermediates + preallocation of workspace
            const bool ifDirect = false;
            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, debug_formulae);
            timing.tf3 = tools::get_time();

            // generation of renormalization block [lc/lr/cr]
            rinter.init(ifDirect, schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            preprocess_formulae_Rlist(ifDirect,superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            timing.tf6 = tools::get_time();

            get_MMlist(Rlst);

            worktot = maxthreads*(blksize*2+qops._size);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            timing.tf7 = tools::get_time();

            preprocess_renorm(qops._data, site._data, size, rank, qops._size, blksize, Rlst, opaddr);
            timing.tf8 = tools::get_time();

         }else if(alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            const bool ifSingle = alg_renorm > 7;
            const bool ifDirect = alg_renorm % 2 == 1;
            if(schd.ctns.alg_rinter == 2){
               std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter!=2" << std::endl;
               exit(1);
            }
            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, debug_formulae);
            timing.tf3 = tools::get_time();

            // generation of renormalization block [lc/lr/cr]
            rinter.init(ifDirect, schd.ctns.alg_rinter, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Rlist2(ifDirect, superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                     Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Rlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Rlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Rlist(ifDirect, superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Rlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0);
            timing.tf6 = tools::get_time();

            // determine batchsize dynamically
            size_t blocksize = 2*blksize+blksize0;
            preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch,
                  batchsize, worktot);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " blksize0=" << blksize0 << " batchsize=" << batchsize
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];

            // generate Rmmtasks
            const int batchblas = schd.ctns.alg_rinter; // use the same keyword for GEMM_batc
            if(!ifSingle){
               Rmmtasks.resize(Rlst2.size());
               for(int i=0; i<Rmmtasks.size(); i++){
                  Rmmtasks[i].init(Rlst2[i], schd.ctns.mmorder, batchblas, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                     std::cout << " rank=" << rank << " iblk=" << i
                        << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                        << " batchsize=" << Rmmtasks[i].batchsize
                        << " nbatch=" << Rmmtasks[i].nbatch
                        << std::endl;
                  }
               }
            }else{
               Rmmtask.init(Rlst, schd.ctns.mmorder, batchblas, batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1){
                  std::cout << " rank=" << rank 
                     << " Rlst.size=" << Rlst.size()
                     << " Rmmtask.totsize=" << Rmmtask.totsize
                     << " batchsize=" << Rmmtask.batchsize 
                     << " nbatch=" << Rmmtask.nbatch 
                     << std::endl;
               }
            }
            timing.tf7 = tools::get_time();

            // initialization of qops
            memset(qops._data, 0, qops._size*sizeof(Tm));
            timing.tf8 = tools::get_time();

            // kernel
            if(!ifSingle){
               if(!ifDirect){ 
                  preprocess_renorm_batch(qops._data, site._data, size, rank, qops._size,
                        Rmmtasks, opaddr, workspace);
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  preprocess_renorm_batchDirect(qops._data, site._data, size, rank, qops._size,
                        Rmmtasks, opaddr, workspace, 
                        rinter._data);
               }
            }else{
               if(!ifDirect){ 
                  preprocess_renorm_batchSingle(qops._data, site._data, size, rank, qops._size,
                        Rmmtask, opaddr, workspace);
               }else{
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  preprocess_renorm_batchDirectSingle(qops._data, site._data, size, rank, qops._size,
                        Rmmtask, opaddr, workspace, 
                        rinter._data);
               }
            }
            timing.tf9 = tools::get_time();
            timing.tf10 = tools::get_time();

#ifdef GPU
         }else if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            const bool ifSingle = alg_renorm > 17;
            const bool ifDirect = alg_renorm % 2 == 1;

            // allocate memery on GPU & copy qops 
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
            qops.allocate_gpu(true);
            size_t opertot = qops1.size() + qops2.size() + qops.size();
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }

            size_t gpumem_site = sizeof(Tm)*site.size();
            dev_site = (Tm*)GPUmem.allocate(gpumem_site);
            GPUmem.to_gpu(dev_site, site._data, gpumem_site);
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,site)(GB)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_site/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, debug_formulae);

            timing.tf3 = tools::get_time();

            // compute hintermediates on CPU
            rinter.init(ifDirect, schd.ctns.alg_rinter, qops_dict, oploc, dev_opaddr, rtasks, debug);
            size_t gpumem_rinter = sizeof(Tm)*rinter.size();
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,site,rinter)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_site/std::pow(1024.0,3) 
                  << "," << gpumem_rinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Rlist2(ifDirect, superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                     Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Rlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Rlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Rlist(ifDirect, superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Rlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 
            timing.tf6 = tools::get_time();

            auto t0y = tools::get_time();
            // Determine batchsize dynamically
            size_t blocksize = 2*blksize+blksize0+1;
            preprocess_gpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch, 0, rank, 
                  batchsize, gpumem_batch);
            dev_workspace = (Tm*)GPUmem.allocate(gpumem_batch);
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,site,rinter,batch)=" << gpumem_oper/std::pow(1024.0,3) 
                  << "," << gpumem_site/std::pow(1024.0,3) 
                  << "," << gpumem_rinter/std::pow(1024.0,3) 
                  << "," << gpumem_batch/std::pow(1024.0,3)
                  << " blksize=" << blksize
                  << " blksize0=" << blksize0 
                  << " batchsize=" << batchsize 
                  << std::endl;
            }
            auto t1y = tools::get_time();

            // generate Rmmtasks given batchsize
            const int batchblas = 2;
            if(!ifSingle){
               Rmmtasks.resize(Rlst2.size());
               for(int i=0; i<Rmmtasks.size(); i++){
                  Rmmtasks[i].init(Rlst2[i], schd.ctns.mmorder, batchblas, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                     std::cout << " rank=" << rank << " iblk=" << i
                        << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                        << " batchsize=" << Rmmtasks[i].batchsize
                        << " nbatch=" << Rmmtasks[i].nbatch
                        << std::endl;
                  }
               }
            }else{
               Rmmtask.init(Rlst, schd.ctns.mmorder, batchblas, batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1){
                  std::cout << " rank=" << rank 
                     << " Rlst.size=" << Rlst.size()
                     << " Rmmtask.totsize=" << Rmmtask.totsize
                     << " batchsize=" << Rmmtask.batchsize 
                     << " nbatch=" << Rmmtask.nbatch 
                     << std::endl;
               }
            }
            auto t2y = tools::get_time();
            if(rank == 0){
               std::cout << "timing: allocate=" 
                  << tools::get_duration(t1y-t0y)
                  << " rmmtasks init="
                  << tools::get_duration(t2y-t1y)
                  << " total="  
                  << tools::get_duration(t2y-t0y)
                  << std::endl;
            }
            timing.tf7 = tools::get_time();
            timing.tf8 = tools::get_time();

            // kernel
            dev_red = dev_workspace + batchsize*(blksize*2+blksize0);
            if(!ifSingle){
               if(!ifDirect){
                  preprocess_renorm_batchGPU(qops._dev_data, dev_site, size, rank, qops._size,
                        Rmmtasks, dev_opaddr, dev_workspace, 
                        dev_red);
               }else{
                  dev_opaddr[4] = dev_workspace + batchsize*blksize*2; // tmpspace for intermediates
                  preprocess_renorm_batchDirectGPU(qops._dev_data, dev_site, size, rank, qops._size,
                        Rmmtasks, dev_opaddr, dev_workspace, 
                        rinter._dev_data, dev_red);
               }
            }else{
               if(!ifDirect){
                  preprocess_renorm_batchGPUSingle(qops._dev_data, dev_site, size, rank, qops._size,
                        Rmmtask, dev_opaddr, dev_workspace, 
                        dev_red);
               }else{
                  dev_opaddr[4] = dev_workspace + batchsize*blksize*2; // tmpspace for intermediates
                  preprocess_renorm_batchDirectGPUSingle(qops._dev_data, dev_site, size, rank, qops._size,
                        Rmmtask, dev_opaddr, dev_workspace, 
                        rinter._dev_data, dev_red);
               }
            }
            timing.tf9 = tools::get_time();

            auto t0x = tools::get_time();
            // copy results back to CPU immediately, if NOT async_tocpu
            if(!schd.ctns.async_tocpu) qops.to_cpu();
            auto t1x = tools::get_time();
            
            timing.tf10 = tools::get_time();

            GPUmem.deallocate(dev_site, gpumem_site);
            auto t2x = tools::get_time();
            GPUmem.deallocate(dev_workspace, gpumem_batch);
            auto t3x = tools::get_time();
            if(rank == 0){
               std::cout << "timing: gpu2cpu=" 
                  << tools::get_duration(t1x-t0x)
                  << " deallocate(site/work)="
                  << tools::get_duration(t2x-t1x)
                  << ","
                  << tools::get_duration(t3x-t1x)
                  << " total="
                  << tools::get_duration(t3x-t0x)
                  << std::endl;
            }

#endif // GPU

         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm
         timing.tf11 = tools::get_time();

         // debug 
         if(debug_oper_renorm && rank == 0){
            const int target = -1;
            std::cout << "\nqops: rank=" << rank << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank
                     << " key=" << key
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
            symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.verbose);
            std::cout << "\nqops[ref]: rank=" << rank << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank
                     << " key=" << key
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
            std::cout << "\nqops[diff]: rank=" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "rank=" << rank 
                     << " key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second[diff]=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == target) pr.second.print("Cdiff",2);
               }
            }
            std::cout << "rank=" << rank << " total diff=" << diff << std::endl;
            linalg::xcopy(qops._size, data0, qops._data);
            delete[] data0;
            delete[] data1;
            if(diff > thresh_opdiff) exit(1);
         }

         // free tmp space on CPU
         if(alg_renorm==6 || alg_renorm==7 || alg_renorm==8 || alg_renorm==9){
            delete[] workspace;
         }

         if(ifdist1 and schd.ctns.async_tocpu){
            std::cout << "error: ifdist1 and async_tocpu are not compatible!" << std::endl;
            exit(1);
         }
         if(!schd.ctns.async_tocpu){
         
            // 2. reduce 
#ifndef SERIAL
            if(ifdist1 and size > 1){
               // Sp[iproc] += \sum_i Sp[i]
               auto opS_index = qops.oper_index_op('S');
               for(int p : opS_index){
                  int iproc = distribute1(ifkr,size,p);
                  auto& opS = qops('S')[p];
                  size_t opsize = opS.size();
                  size_t off = qops._offset[std::make_pair('S',p)];
                  mpi_wrapper::reduce(icomb.world, opS.data(), opsize, iproc, schd.ctns.alg_comm);
                  if(iproc == rank){ 
#ifdef GPU
                     if(schd.ctns.alg_renorm>10) GPUmem.to_gpu(qops._dev_data+off, opS.data(), opsize*sizeof(Tm));
#endif
                  }else{
                     opS.set_zero();
#ifdef GPU
                     if(schd.ctns.alg_renorm>10) GPUmem.memset(qops._dev_data+off, opsize*sizeof(Tm));
#endif
                  }
               }
               // H[0] += \sum_i H[i]
               auto& opH = qops('H')[0];
               size_t opsize = opH.size();
               size_t off = qops._offset[std::make_pair('H',0)];
               mpi_wrapper::reduce(icomb.world, opH.data(), opsize, 0, schd.ctns.alg_comm);
               if(rank == 0){ 
#ifdef GPU
                  if(schd.ctns.alg_renorm>10) GPUmem.to_gpu(qops._dev_data+off, opH.data(), opsize*sizeof(Tm));
#endif
               }else{
                  opH.set_zero();
#ifdef GPU
                  if(schd.ctns.alg_renorm>10) GPUmem.memset(qops._dev_data+off, opsize*sizeof(Tm));
#endif
               }
            }
#endif

            // 3. consistency check for Hamiltonian
            const auto& opH = qops('H').at(0);
            auto diffH = (opH-opH.H()).normF();
            if(debug){
               std::cout << "check ||H-H.dagger||=" << std::scientific << std::setprecision(3) << diffH 
                  << " coord=" << p << " rank=" << rank << std::defaultfloat << std::endl;
            } 
            if(diffH > thresh_opdiff){
               std::cout <<  "error in oper_renorm: ||H-H.dagger||=" << std::scientific << std::setprecision(3) << diffH 
                  << " is larger than thresh_opdiff=" << thresh_opdiff 
                  << " for rank=" << rank 
                  << std::endl;
               exit(1);
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
         } // async_tocpu

         timing.tf12 = tools::get_time();
         if(debug){
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
            double t_tot = tools::get_duration(timing.tf12-timing.tf0); 
            double t_init = tools::get_duration(timing.tf1-timing.tf0);
            double t_kernel = tools::get_duration(timing.tf11-timing.tf1);
            double t_comm = tools::get_duration(timing.tf12-timing.tf11);
            std::cout << "----- TIMING FOR oper_renorm : " << t_tot << " S" 
               << " T(init/kernel/comm)=" << t_init << "," << t_kernel << "," << t_comm
               << " rank=" << rank << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
