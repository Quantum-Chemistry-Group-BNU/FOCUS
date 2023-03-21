#ifndef OPER_RENORM_H
#define OPER_RENORM_H

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

namespace ctns{

   const bool debug_oper_renorm = false;
   extern const bool debug_oper_renorm;

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
            const std::string fname){
         int size = 1, rank = 0, maxthreads = 1;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const int isym = Km::isym;
         const bool ifkr = Km::ifkr;
         const int& alg_renorm = schd.ctns.alg_renorm;
         const bool& sort_formulae = schd.ctns.sort_formulae;
         const bool& ifdist1 = schd.ctns.ifdist1;
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
         auto ti = tools::get_time();

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

         /*
         Tm* data0 = new Tm[qops._size];
         memset(data0, 0, qops._size*sizeof(Tm));
         Tm* data1 = new Tm[qops._size];
         memset(data1, 0, qops._size*sizeof(Tm));
         */

         //-------------------------------
         // 1. kernel for renormalization
         //-------------------------------
         oper_timer.start();
         const bool debug_formulae = schd.ctns.verbose>0;
         size_t worktot = 0;
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
         size_t blksize;
         double cost;
         Tm* workspace;
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
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            
            // generation of renormalization block [lc/lr/cr]
            rinter.init(schd.ctns.alg_inter, qops_dict, oploc, opaddr, rtasks, debug);

            // GEMM list and GEMV list
            preprocess_formulae_Rlist(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst, blksize, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist(Rlst, schd.ctns.hxorder);

            worktot = maxthreads*(blksize*2+qops._size);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for renorm: size=" << qops._size << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }

            /*
            // oldest version
            auto rfuns = oper_renorm_functors(superblock, site, int2e, qops1, qops2, qops, ifdist1);
            oper_renorm_kernel(superblock, rfuns, site, qops, schd.ctns.verbose);
            linalg::xcopy(qops._size, qops._data, data0);

            std::cout << "\nlzd:qops: old" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               //if(key != 'C') continue;
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == 0) pr.second.print("C0old",2);
                  //if(pr.first == 9) pr.second.print("C8old",2);
               }
            }
            memset(qops._data, 0, qops._size*sizeof(Tm));
            */
            
            preprocess_renorm(qops._data, site._data, size, rank, qops._size, blksize, Rlst, opaddr);
            /* 
            linalg::xcopy(qops._size, qops._data, data1);
          
            std::cout << "\nlzd:qops: new" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               //if(key != 'C') continue;
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
                  if(key == 'C' and pr.first == 0) pr.second.print("C0new",2);
               }
            }
            linalg::xaxpy(qops._size, -1.0, data0, qops._data);
            auto diff = linalg::xnrm2(qops._size, qops._data);

            std::cout << "\nlzd:qops-diff" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               //if(key != 'C') continue;
               for(auto& pr : opdict){
                  if(pr.second.normF() < 1.e-10) continue;
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
                  //if(pr.first == 9) pr.second.print("C8diff",2);
                  if(key == 'C' and pr.first == 0) pr.second.print("C0diff",2);
               }
            }
            std::cout << "total diff=" << diff << std::endl;
            if(diff > 1.e-10) exit(1);
            linalg::xcopy(qops._size, data1, qops._data);
            */

         }else if(alg_renorm == 6){

            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }

            // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
            auto rtasks = symbolic_formulae_renorm(superblock, int2e, qops1, qops2, qops, 
                  size, rank, fname, sort_formulae, ifdist1,
                  debug_formulae);
            
            // generation of renormalization block [lc/lr/cr]
            rinter.init(schd.ctns.alg_inter, qops_dict, oploc, opaddr, rtasks, debug);

            // GEMM list and GEMV list
            preprocess_formulae_Rlist2(superblock, qops, qops_dict, oploc, rtasks, site, rinter,
                  Rlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Rlst2.size(); i++){
               maxbatch = std::max(maxbatch, Rlst2[i].size());
            } // i
            size_t batchsize = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;
 
            // generate Rmmtasks
            Rmmtasks.resize(Rlst2.size());
            for(int i=0; i<Rmmtasks.size(); i++){
               Rmmtasks[i].init(Rlst2[i], schd.ctns.batchblas, batchsize,
                     blksize*2, schd.ctns.hxorder);
               if(debug && schd.ctns.verbose>1){
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
            workspace = new Tm[worktot];

            /*
            // oldest version
            auto rfuns = oper_renorm_functors(superblock, site, int2e, qops1, qops2, qops, ifdist1);
            oper_renorm_kernel(superblock, rfuns, site, qops, schd.ctns.verbose);
            linalg::xcopy(qops._size, qops._data, data0);

            std::cout << "\nlzd:qops: old" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " size=" << pr.second.size()
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
               }
            }
            memset(qops._data, 0, qops._size*sizeof(Tm));
            */
           
            preprocess_renorm_batch(qops._data, site._data, size, rank, qops._size, blksize, Rlst2, 
                                    Rmmtasks, opaddr, workspace);
            delete[] workspace;
            /*
            linalg::xcopy(qops._size, qops._data, data1);

            std::cout << "\nlzd:qops: new" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  std::cout << "key=" << key
                     << " pr.first=" << pr.first
                     << " size=" << pr.second.size()
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
               }
            }
            linalg::xaxpy(qops._size, -1.0, data0, qops._data);
            auto diff = linalg::xnrm2(qops._size, qops._data);

            std::cout << "\nlzd:qops-diff" << std::endl;
            for(auto& key : qops.oplist){
               auto& opdict = qops(key);
               for(auto& pr : opdict){
                  if(pr.second.normF() < 1.e-10) continue;
                  std::cout << ">>> key=" << key
                     << " pr.first=" << pr.first
                     << " size=" << pr.second.size()
                     << " pr.second=" << pr.second.normF()
                     << std::endl;
               }
            }
            std::cout << "total diff=" << diff << std::endl;
            if(diff > 1.e-10) exit(1);
            linalg::xcopy(qops._size, data1, qops._data);
            */

         }else if(alg_renorm == 7){

            // BatchGPU: symbolic formulae + rintermediates + preallocation of workspace
            std::cout << "Not implemented yet!" << std::endl;
            exit(1);

         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         }
         //-------------------------------

         // 2. reduce 
         auto ta = tools::get_time();
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
         if(diffH > 1.e-10){
            opH.print("H",2);
            std::string msg = "error: H-H.H() is too large! diffH=";
            tools::exit(msg+std::to_string(diffH));
         }

         // check against explicit construction
         if(debug_oper_renorm){
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

         auto tf = tools::get_time();
         if(debug){ 
            if(schd.ctns.verbose>0) qops.print("qops");
            if(alg_renorm == 0 && schd.ctns.verbose>1) oper_timer.analysis();
            double t_tot = tools::get_duration(tf-ti); 
            double t_cal = tools::get_duration(ta-ti);
            double t_comm = tools::get_duration(tf-ta);
            std::cout << "TIMING for Renormalization : " << t_tot 
               << "  T(cal/comm)=" << t_cal << "," << t_comm
               << std::endl;
         }
         return worktot;
      }

} // ctns

#endif
