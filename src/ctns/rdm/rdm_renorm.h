#ifndef RDM_RENORM_H
#define RDM_RENORM_H

/*
#include <type_traits>
#include "ctns_sys.h"
#include "sweep_data.h"
#include "oper_timer.h"
#include "oper_functors.h"
#include "oper_normxwf.h"
#include "oper_compxwf.h"
#include "oper_rbasis.h"
#include "oper_renorm_kernel.h"
#include "symbolic_formulae_renorm.h"
#include "sadmrg/symbolic_formulae_renorm_su2.h"
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
*/

namespace ctns{

   // renormalize operators
   template <typename Qm, typename Tm>
      void rdm_renorm(const int order,
            const std::string superblock,
            const bool is_same,
            const comb<Qm,Tm>& icomb,
            const comb<Qm,Tm>& icomb2,
            const comb_coord& p,
            const input::schedule& schd,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const std::string fname,
            dot_timing& timing,
            const std::string fmmtask=""){
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const int sorb = icomb.get_nphysical()*2;
         const int alg_renorm = schd.ctns.alg_renorm;
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool sort_formulae = schd.ctns.sort_formulae;
         const bool ifdist1 = schd.ctns.ifdist1;
         const bool ifdistc = schd.ctns.ifdistc;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::rdm_renorm coord=" << p 
               << " superblock=" << superblock 
               << " ifab=" << Qm::ifabelian
               << " isym=" << isym 
               << " ifkr=" << ifkr
               << " is_same=" << is_same
               << " sorb=" << sorb
               << " alg_renorm=" << alg_renorm	
               << " mpisize=" << size
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         timing.tf0 = tools::get_time(); 

         // 0. setup basic information for qops
         qops.sorb = sorb;
         qops.isym = isym;
         qops.ifkr = ifkr;
         qops.cindex = oper_combine_cindex(qops1.cindex, qops2.cindex);
         // rest of spatial orbital indices
         const auto& node = icomb.topo.get_node(p);
         const auto& rindex = icomb.topo.rindex;
         const auto& site = icomb.sites[rindex.at(p)];
         const auto& site2 = icomb2.sites[rindex.at(p)];
         if(superblock == "cr"){
            //  ---*--- site
            //     |   \
            //     *    *
            //     |   /
            //  ---*--- site2
            qops.krest = node.lorbs;
            qops.qbra = site.info.qrow;
            qops.qket = site2.info.qrow;
            assert(check_consistency(site.info.qmid, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
            assert(check_consistency(site2.info.qmid, qops1.qket));
            assert(check_consistency(site2.info.qcol, qops2.qket));
         }else if(superblock == "lc"){
            qops.krest = node.rorbs;
            qops.qbra = site.info.qcol;
            qops.qket = site2.info.qcol;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qmid, qops2.qbra));
            assert(check_consistency(site2.info.qrow, qops1.qket));
            assert(check_consistency(site2.info.qmid, qops2.qket));
         }else if(superblock == "lr"){
            qops.krest = node.corbs;
            qops.qbra = site.info.qmid;
            qops.qket = site2.info.qmid;
            assert(check_consistency(site.info.qrow, qops1.qbra));
            assert(check_consistency(site.info.qcol, qops2.qbra));
            assert(check_consistency(site2.info.qrow, qops1.qket));
            assert(check_consistency(site2.info.qcol, qops2.qket));
            tools::exit("error: rdm_renorm does not support superblock=lr yet!");
         }
         if(order == 1){ 
            if(is_same){
               qops.oplist = "C";
            }else{
               qops.oplist = "C";
               //std::cout << "error: not implemented yet";
               //exit(1);
            }
         }else if(order == 2){
            if(is_same){
               qops.oplist = "CAB";
            }else{
               std::cout << "error: not implemented yet";
               //qops.oplist = "CABD";
               exit(1);
            }
         }else{
            std::cout << "error: rdm_renorm does not support order=" << order << std::endl;
            exit(1);
         }
         qops.mpisize = size;
         qops.mpirank = rank;
         qops.ifdist2 = true;
         // initialize
         qops.init();
         if(debug){ 
            qops.print("qops");
            get_sys_status();
         }

         //-------------------------------
         // 1. kernel for renormalization
         //-------------------------------
         oper_timer.dot_start();
         const bool debug_formulae = schd.ctns.verbose>0;
         size_t worktot=0;
         // intermediates
         rintermediates<Qm::ifabelian,Tm> rinter;
         Rlist<Tm> Rlst;
         Rlist2<Tm> Rlst2;
         RMMtask<Tm> Rmmtask;
         RMMtasks<Tm> Rmmtasks;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const qoper_dictmap<Qm::ifabelian,Tm> qops_dict = {{block1,qops1}, {block2,qops2}};
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
         Tm* workspace = nullptr;
#ifdef GPU
         Tm* dev_site = nullptr;
         Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
         Tm* dev_workspace = nullptr;
         Tm* dev_red = nullptr;
#endif
         size_t batchsize, gpumem_batch;

         // consistency check
         if(schd.ctns.ifdistc && !icomb.topo.ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }
         if(Qm::ifabelian && Qm::ifkr && alg_renorm >=4){
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

         // declare a fake int2e
         integral::two_body<Tm> int2e;
         if(alg_renorm == 0){
/*

            // oldest version
            auto rfuns = oper_renorm_functors(superblock, site2, int2e, qops1, qops2, qops, ifdist1);
            // initialize of qops
            oper_renorm_kernel(superblock, rfuns, site, qops, schd.ctns.verbose);

         }else if(alg_renorm == 1){

            // symbolic formulae + dynamic allocation of memory
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, ifdistc, debug_formulae);
            // initialization of qops inside
            symbolic_kernel_renorm(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.ifdist1, schd.ctns.verbose);

         }else if(alg_renorm == 2){

            // symbolic formulae + preallocation of workspace
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, ifdistc, debug_formulae);
            // initialization of qops inside 
            symbolic_kernel_renorm2(superblock, rtasks, site, qops1, qops2, qops, schd.ctns.ifdist1, schd.ctns.verbose);

         }else if(alg_renorm == 4){

            // CPU: symbolic formulae + rintermediates + preallocation of workspace

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, ifdistc, debug_formulae);

            // generation of renormalization block [lc/lr/cr]
            const bool ifDirect = false;
            const int batchgemv = 1;
            rinter.init(ifDirect, schd.ctns.alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, debug);

            // GEMM list and GEMV list
            preprocess_formulae_Rlist(ifDirect, schd.ctns.alg_rcoper, superblock, 
                  qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
                  Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist2(Rlst);

            worktot = maxthreads*(blksize*2+qops._size);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            // initialization of qops inside
            preprocess_renorm(qops._data, site._data, size, rank, qops._size, blksize, Rlst, opaddr);

         }else if(alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            if(schd.ctns.alg_rinter == 2){
               std::cout << "error: alg_renorm=" << alg_renorm << " should be used with alg_rinter!=2" << std::endl;
               exit(1);
            }
            timing.tf1 = tools::get_time();
            timing.tf2 = tools::get_time();

            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1, ifdistc, debug_formulae);
            timing.tf3 = tools::get_time();

            // generation of renormalization block [lc/lr/cr]
            const bool ifSingle = alg_renorm > 7;
            const bool ifDirect = alg_renorm % 2 == 1;
            const int batchgemv = 1;
            rinter.init(ifDirect, schd.ctns.alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, debug);
            timing.tf4 = tools::get_time();
            timing.tf5 = tools::get_time();

            // GEMM list and GEMV list
            size_t maxbatch = 0;
            if(!ifSingle){
               preprocess_formulae_Rlist2(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
                     Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Rlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Rlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Rlist(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Rlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0);
            timing.tf6 = tools::get_time();

            if(blksize > 0){
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
               const int batchblas = schd.ctns.alg_rinter; // use the same keyword for GEMM_batch
               auto batchrenorm = std::make_tuple(batchblas,batchblas,batchblas); 
               if(!ifSingle){
                  Rmmtasks.resize(Rlst2.size());
                  for(int i=0; i<Rmmtasks.size(); i++){
                     Rmmtasks[i].init(Rlst2[i], batchblas, batchrenorm, batchsize, blksize*2, blksize0);
                     if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                        std::cout << " rank=" << rank << " iblk=" << i
                           << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                           << " batchsize=" << Rmmtasks[i].batchsize
                           << " nbatch=" << Rmmtasks[i].nbatch
                           << std::endl;
                     }
                  }
                  if(fmmtask.size()>0) save_mmtask(Rmmtasks, fmmtask);
               }else{
                  Rmmtask.init(Rlst, batchblas, batchrenorm, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1){
                     std::cout << " rank=" << rank 
                        << " Rlst.size=" << Rlst.size()
                        << " Rmmtask.totsize=" << Rmmtask.totsize
                        << " batchsize=" << Rmmtask.batchsize 
                        << " nbatch=" << Rmmtask.nbatch 
                        << std::endl;
                  }
                  if(fmmtask.size()>0) save_mmtask(Rmmtask, fmmtask);
               }
            } // blksize>0
            timing.tf7 = tools::get_time();

            // initialization of qops
            memset(qops._data, 0, qops._size*sizeof(Tm));
            timing.tf8 = tools::get_time();

            // kernel
            if(blksize > 0){
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
            } // blksize>0
            timing.tf9 = tools::get_time();
            timing.tf10 = tools::get_time();
            timing.tf11 = tools::get_time();

#ifdef GPU
         }else if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace

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
            // initialization of qops
            qops.allocate_gpu(true);
            size_t opertot = qops1.size() + qops2.size() + qops.size();
            size_t gpumem_oper = sizeof(Tm)*opertot;
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem(GB): used=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper)=" << gpumem_oper/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tf1 = tools::get_time();

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
                  size, rank, fname, sort_formulae, ifdist1, ifdistc, debug_formulae);

            timing.tf3 = tools::get_time();

            // compute hintermediates on CPU
            const bool ifSingle = alg_renorm > 17;
            const bool ifDirect = alg_renorm % 2 == 1;
            const int batchgemv = std::get<0>(schd.ctns.batchrenorm);
            rinter.init(ifDirect, schd.ctns.alg_rinter, batchgemv, qops_dict, oploc, dev_opaddr, rtasks, debug);
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
               preprocess_formulae_Rlist2(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
                     Rlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               for(int i=0; i<Rlst2.size(); i++){
                  maxbatch = std::max(maxbatch, Rlst2[i].size());
               } // i
            }else{
               preprocess_formulae_Rlist(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     qops, qops_dict, oploc, opaddr, rtasks, site, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               maxbatch = Rlst.size();
            }
            if(!ifDirect) assert(blksize0 == 0); 
            timing.tf6 = tools::get_time();

            if(blksize > 0){
               auto t0y = tools::get_time();
               // Determine batchsize dynamically & GPUmem.allocate
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
               const int batchblas = 2; // GPU
               if(!ifSingle){
                  Rmmtasks.resize(Rlst2.size());
                  for(int i=0; i<Rmmtasks.size(); i++){
                     Rmmtasks[i].init(Rlst2[i], batchblas, schd.ctns.batchrenorm, batchsize, blksize*2, blksize0);
                     if(debug && schd.ctns.verbose>1 && Rlst2[i].size()>0){
                        std::cout << " rank=" << rank << " iblk=" << i
                           << " Rmmtasks.totsize=" << Rmmtasks[i].totsize
                           << " batchsize=" << Rmmtasks[i].batchsize
                           << " nbatch=" << Rmmtasks[i].nbatch
                           << std::endl;
                     }
                  } // i
                  if(fmmtask.size()>0) save_mmtask(Rmmtasks, fmmtask);
               }else{
                  Rmmtask.init(Rlst, batchblas, schd.ctns.batchrenorm, batchsize, blksize*2, blksize0);
                  if(debug && schd.ctns.verbose>1){
                     std::cout << " rank=" << rank 
                        << " Rlst.size=" << Rlst.size()
                        << " Rmmtask.totsize=" << Rmmtask.totsize
                        << " batchsize=" << Rmmtask.batchsize 
                        << " nbatch=" << Rmmtask.nbatch 
                        << std::endl;
                     if(schd.ctns.verbose>2){
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
            } // blksize>0
            timing.tf7 = tools::get_time();
            timing.tf8 = tools::get_time();

            // kernel
            if(blksize > 0){
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
            } // blksize>0
            timing.tf9 = tools::get_time();

#ifndef SERIAL
            if(ifdist1 and size > 1 and schd.ctns.ifnccl){
#ifndef NCCL
               std::cout << "error: NCCL must be used for comm[opS,opH] for ifnccl=true!" << std::endl;
               exit(1);
#else
               // Use NCCL to perform reduction for opS and opH on GPU directly
               // Sp[iproc] += \sum_i Sp[i]
               auto opS_index = qops.oper_index_op('S');
               for(int p : opS_index){
                  int iproc = distribute1(ifkr,size,p);
                  auto& opS = qops('S')[p];
                  size_t opsize = opS.size();
                  size_t off = qops._offset[std::make_pair('S',p)];
                  Tm* dev_ptr = qops._dev_data+off;
                  nccl_comm.reduce(dev_ptr, opsize, iproc);
                  if(iproc != rank) GPUmem.memset(dev_ptr, opsize*sizeof(Tm));
               }
               // H[0] += \sum_i H[i]
               auto& opH = qops('H')[0];
               size_t opsize = opH.size();
               size_t off = qops._offset[std::make_pair('H',0)];
               Tm* dev_ptr = qops._dev_data+off;
               nccl_comm.reduce(dev_ptr, opsize, 0);
               if(rank != 0) GPUmem.memset(dev_ptr, opsize*sizeof(Tm));
#endif
            }
#endif // SERIAL
            timing.tf10 = tools::get_time();
            
            qops.to_cpu();
            timing.tf11 = tools::get_time();
            
            GPUmem.deallocate(dev_site, gpumem_site);
            if(blksize > 0) GPUmem.deallocate(dev_workspace, gpumem_batch);
            
            auto tf = tools::get_time();
            if(rank == 0){
               std::cout << "timing: comm[opS,opH]=" << tools::get_duration(timing.tf10-timing.tf9)
                  << " gpu2cpu=" << tools::get_duration(timing.tf11-timing.tf10)
                  << " deallocate(site,work)=" << tools::get_duration(tf-timing.tf11)
                  << " T(tot)=" << tools::get_duration(tf-timing.tf9)
                  << std::endl;
            }
#endif // GPU
*/ 
         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm
         timing.tf12 = tools::get_time();

         // free tmp space on CPU
         if(alg_renorm==6 || alg_renorm==7 || alg_renorm==8 || alg_renorm==9){
            delete[] workspace;
         }

         timing.tf13 = tools::get_time();
         if(debug){
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
            double t_tot = tools::get_duration(timing.tf13-timing.tf0); 
            double t_init = tools::get_duration(timing.tf1-timing.tf0);
            double t_kernel = tools::get_duration(timing.tf12-timing.tf1);
            double t_comm = tools::get_duration(timing.tf13-timing.tf12);
            std::cout << "----- TIMING FOR rdm_renorm : " << t_tot << " S" 
               << " T(init/kernel/comm)=" << t_init << "," << t_kernel << "," << t_comm
               << " rank=" << rank << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
