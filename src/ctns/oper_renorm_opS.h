#ifndef OPER_RENORM_OPS_H
#define OPER_RENORM_OPS_H

#include "symbolic_compxwf.h"
#include "sadmrg/symbolic_compxwf_su2.h"
#include "preprocess_rlist.h"
#include "preprocess_rinter.h"
#include "preprocess_rmu.h"
#include "preprocess_rmmtask.h"

namespace ctns{

#ifndef SERIAL

   // ZL@2024/12/22 opS
   // alg_renorm = 0
   template <typename Qm, typename Tm>
      void oper_renorm_kernel_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3su2<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         std::cout << "error: no implementation of oper_renorm_kernel_opS for su2!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm>
      void oper_renorm_kernel_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         auto sindex = oper_index_opS(qops.krest, qops.ifkr);
         for(const auto& index : sindex){
            if(rank==0 and schd.ctns.verbose>2) std::cout << " opS: index=" << index << std::endl;
            // compute opS[p](iproc) 
            auto opxwf = oper_compxwf_opS(superblock, site, qops1, qops2, int2e, index, size, rank, schd.ctns.ifdist1);
            auto op = contract_qt3_qt3(superblock, site, opxwf);
            // reduction of op
            int iproc = distribute1(qops.ifkr, size, index);
            mpi_wrapper::reduce(icomb.world, op.data(), op.size(), iproc);
            if(iproc == rank){
               auto& opS = qops('S')[index];
               linalg::xcopy(op.size(), op.data(), opS.data());
            }
         }
      }

   // alg_renorm = 1
   template <typename Qm, typename Tm>
      void symbolic_kernel_renorm_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3su2<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         std::cout << "error: no implementation of symbolic_kernel_renorm_opS for su2!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm>
      void symbolic_kernel_renorm_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const stensor3<Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         const char key = 'S';
         const bool skipId = true;
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const oper_dictmap<Tm> qops_dict = {{block1,qops1},
            {block2,qops2}};
         auto sindex = oper_index_opS(qops.krest, qops.ifkr);
         for(const auto& index : sindex){
            if(rank==0 and schd.ctns.verbose>2) std::cout << " opS: index=" << index << std::endl;
            auto sym_op = get_qsym_opS(Qm::isym, index); 
            stensor2<Tm> op(sym_op, qops.qbra, qops.qket);
            // generate formula for opS[p]
            auto formula = symbolic_compxwf_opS<Tm>(qops1.oplist, qops2.oplist, 
                  block1, block2, qops1.cindex, qops2.cindex,
                  int2e, index, Qm::isym, Qm::ifkr, size, rank, schd.ctns.ifdist1, schd.ctns.ifdistc);
            // opS can be empty for ifdist1=true
            if(formula.size() != 0){
               auto opxwf = symbolic_renorm_single(block1, block2, qops_dict, key, formula, site, skipId);
               auto optmp = contract_qt3_qt3(superblock, site, opxwf);
               linalg::xcopy(optmp.size(), optmp.data(), op.data());
            }
            // reduction of op
            int iproc = distribute1(qops.ifkr, size, index);
            mpi_wrapper::reduce(icomb.world, op.data(), op.size(), iproc);
            if(iproc == rank){
               auto& opS = qops('S')[index];
               linalg::xcopy(op.size(), op.data(), opS.data());
            }
         }
      }

   // alg_renorm = 4
   template <bool ifab, typename Tm, typename QTm>
      void preprocess_formulae_Rlist_opS(const bool ifDirect,
            const int alg_rcoper,
            const std::string superblock,
            const qtensor2<ifab,Tm>& opSp,
            const qoper_dictmap<ifab,Tm>& qops_dict,
            const std::map<std::string,int>& oploc,
            Tm** opaddr,
            const renorm_tasks<Tm>& rtasks,
            const QTm& site,
            const QTm& site2,
            const bool skipId,
            const rintermediates<ifab,Tm>& rinter,
            Rlist<Tm>& Rlst,
            size_t& blksize,
            size_t& blksize0,
            double& cost,
            const bool debug){
         auto t0 = tools::get_time();

         // 1. preprocess formulae to Rmu
         int rsize = rtasks.size();
         std::vector<std::vector<Rmu_ptr<ifab,Tm>>> Rmu(rsize);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif	
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            Rmu[k].resize(formula.size());
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].rinfo = const_cast<qinfo2type<ifab,Tm>*>(&opSp.info);
               Rmu[k][it].offrop = 0;
               Rmu[k][it].init(ifDirect, k, it, formula, qops_dict, rinter, oploc, skipId);
            }
         } // it

         // 2. from Rmu to expanded block forms
         blksize = 0;
         blksize0 = 0;
         cost = 0.0;
         int nnzblk = opSp.info.qrow.size()*opSp.info.qcol.size(); // partitioned according to (rows,cols)
         Rlist2<Tm> Rlst2;
         Rlst2.resize(nnzblk);
         for(int k=0; k<rsize; k++){
            const auto& task = rtasks.op_tasks[k];
            const auto& key = std::get<0>(task);
            const auto& index = std::get<1>(task);
            const auto& formula = std::get<2>(task);
            for(int it=0; it<formula.size(); it++){
               Rmu[k][it].gen_Rlist2(alg_rcoper, opaddr, superblock, site.info, site2.info, Rlst2, blksize, blksize0, cost, false);
            }
         }

         // 3. copy Rlist2 to Rlst
         size_t size = 0;
         for(int i=0; i<Rlst2.size(); i++){
            size += Rlst2[i].size();
         }
         Rlst.reserve(size);
         for(int i=0; i<Rlst2.size(); i++){
            std::move(Rlst2[i].begin(), Rlst2[i].end(), std::back_inserter(Rlst));
         }

         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR preprocess_formulae_Rlist_opS : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Qm, typename Tm>
      void preprocess_renorm_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const qtensor3<Qm::ifabelian,Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         int maxthreads = 1;
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const char key = 'S';
         const bool skipId = true;
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const qoper_dictmap<Qm::ifabelian,Tm> qops_dict = {{block1,qops1}, {block2,qops2}};
         const std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
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
         auto sindex = oper_index_opS(qops.krest, qops.ifkr);
         for(const auto& index : sindex){
            if(rank==0 and schd.ctns.verbose>2) std::cout << " opS: index=" << index << std::endl;
            auto sym_op = get_qsym_opS(Qm::isym, index); 
            qtensor2<Qm::ifabelian,Tm> op(sym_op, qops.qbra, qops.qket);
            // generate formula for opS[p]
            symbolic_task<Tm> formula;
            if(Qm::ifabelian){
               formula = symbolic_compxwf_opS<Tm>(qops1.oplist, qops2.oplist, 
                     block1, block2, qops1.cindex, qops2.cindex,
                     int2e, index, Qm::isym, Qm::ifkr, size, rank, schd.ctns.ifdist1, schd.ctns.ifdistc);
            }else{
               formula = symbolic_compxwf_opS_su2<Tm>(qops1.oplist, qops2.oplist, 
                     block1, block2, qops1.cindex, qops2.cindex,
                     int2e, index, Qm::ifkr, size, rank, schd.ctns.ifdist1, schd.ctns.ifdistc);
            }
            // opS can be empty for ifdist1=true
            if(formula.size() != 0){
               if(rank==0 && schd.ctns.verbose>0){
                  formula.display("opS_rank"+std::to_string(rank)+"_index"+std::to_string(index));
               }
               renorm_tasks<Tm> rtasks;
               rtasks.append(std::make_tuple('S', index, formula));
               // CPU: symbolic formulae + rintermediates + preallocation of workspace
               rintermediates<Qm::ifabelian,Tm> rinter;
               const bool ifDirect = false;
               const int batchgemv = 1;
               rinter.init(ifDirect, schd.ctns.alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, rank==0);
               // GEMM list and GEMV list
               Rlist<Tm> Rlst; 
               size_t blksize=0, blksize0=0;
               double cost=0.0;
               preprocess_formulae_Rlist_opS(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     op, qops_dict, oploc, opaddr, rtasks, site, site, skipId, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               get_MMlist2(Rlst);
               size_t worktot = maxthreads*blksize*2;
               if(rank==0 && schd.ctns.verbose>0){
                  std::cout << "preprocess for renorm_opS: index=" << index << " size=" << op.size() << " blksize=" << blksize 
                     << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                     << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
               }
               // initialization of qops inside
               preprocess_renorm(op.data(), site._data, site._data, size, rank, op.size(), blksize, Rlst, opaddr);
            }
            // reduction of op
            int iproc = distribute1(qops.ifkr, size, index);
            mpi_wrapper::reduce(icomb.world, op.data(), op.size(), iproc);
            if(iproc == rank){
               auto& opS = qops('S')[index];
               linalg::xcopy(op.size(), op.data(), opS.data());
            }
         }
      }

   // alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9
   template <typename Qm, typename Tm>
      void preprocess_renorm_batch_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const qtensor3<Qm::ifabelian,Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops,
            const int size,
            const int rank){
         const int alg_renorm = schd.ctns.alg_renorm;
         const int alg_rinter = schd.ctns.alg_rinter;
         const char key = 'S';
         const bool skipId = true;
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const qoper_dictmap<Qm::ifabelian,Tm> qops_dict = {{block1,qops1}, {block2,qops2}};
         const std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c",2}};
         Tm* opaddr[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
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
         auto sindex = oper_index_opS(qops.krest, qops.ifkr);
         for(const auto& index : sindex){
            if(rank==0 and schd.ctns.verbose>2) std::cout << " opS: index=" << index << std::endl;
            auto sym_op = get_qsym_opS(Qm::isym, index); 
            qtensor2<Qm::ifabelian,Tm> op(sym_op, qops.qbra, qops.qket);
            // generate formula for opS[p]
            symbolic_task<Tm> formula;
            if(Qm::ifabelian){
               formula = symbolic_compxwf_opS<Tm>(qops1.oplist, qops2.oplist, 
                     block1, block2, qops1.cindex, qops2.cindex,
                     int2e, index, Qm::isym, Qm::ifkr, size, rank, schd.ctns.ifdist1, schd.ctns.ifdistc);
            }else{
               formula = symbolic_compxwf_opS_su2<Tm>(qops1.oplist, qops2.oplist, 
                     block1, block2, qops1.cindex, qops2.cindex,
                     int2e, index, Qm::ifkr, size, rank, schd.ctns.ifdist1, schd.ctns.ifdistc);
            }
            // opS can be empty for ifdist1=true
            if(formula.size() != 0){
               if(rank==0 && schd.ctns.verbose>0){
                  formula.display("opS_rank"+std::to_string(rank)+"_index"+std::to_string(index));
               }
               renorm_tasks<Tm> rtasks;
               rtasks.append(std::make_tuple('S', index, formula));
               // BatchCPU: symbolic formulae + rintermediates + preallocation of workspace
               rintermediates<Qm::ifabelian,Tm> rinter;
               const bool ifSingle = true;
               const bool ifDirect = true;
               const int batchgemv = 1;
               rinter.init(ifDirect, alg_rinter, batchgemv, qops_dict, oploc, opaddr, rtasks, rank==0);
               // GEMM list and GEMV list
               Rlist<Tm> Rlst; 
               size_t blksize=0, blksize0=0;
               double cost=0.0;
               preprocess_formulae_Rlist_opS(ifDirect, schd.ctns.alg_rcoper, superblock, 
                     op, qops_dict, oploc, opaddr, rtasks, site, site, skipId, rinter,
                     Rlst, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
               if(blksize > 0){
                  // determine batchsize dynamically
                  size_t blocksize = 2*blksize+blksize0;
                  size_t maxbatch = Rlst.size();
                  size_t batchsize, worktot;
                  preprocess_cpu_batchsize<Tm>(schd.ctns.batchmem, blocksize, maxbatch,
                        batchsize, worktot);
                  if(rank==0 && schd.ctns.verbose>0){
                     std::cout << "preprocess for renorm_opS: size=" << qops._size << " blksize=" << blksize 
                        << " blksize0=" << blksize0 << " batchsize=" << batchsize
                        << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                        << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
                  }
                  Tm* workspace = new Tm[worktot];
                  // generate Rmmtasks
                  const int batchblas = alg_rinter; // use the same keyword for GEMM_batch
                  auto batchrenorm = std::make_tuple(batchblas,batchblas,batchblas);
                  RMMtask<Tm> Rmmtask;
                  Rmmtask.init(Rlst, batchblas, batchrenorm, batchsize, blksize*2, blksize0);
                  if(rank==0 && schd.ctns.verbose>1){
                     std::cout << " rank=" << rank 
                        << " Rlst.size=" << Rlst.size()
                        << " Rmmtask.totsize=" << Rmmtask.totsize
                        << " batchsize=" << Rmmtask.batchsize 
                        << " nbatch=" << Rmmtask.nbatch 
                        << std::endl;
                  }
                  opaddr[4] = workspace + batchsize*blksize*2; // memory layout [workspace|inter]
                  preprocess_renorm_batchDirectSingle(op.data(), site._data, site._data, size, rank, op.size(),
                        Rmmtask, opaddr, workspace, rinter._data);
                  delete[] workspace;
               } // blksize>0
            }
            // reduction of op
            int iproc = distribute1(qops.ifkr, size, index);
            mpi_wrapper::reduce(icomb.world, op.data(), op.size(), iproc);
            if(iproc == rank){
               auto& opS = qops('S')[index];
               linalg::xcopy(op.size(), op.data(), opS.data());
            }
         }
      }

   // driver (following sweep_renorm)
   template <typename Qm, typename Tm>
      void oper_renorm_opS(const std::string superblock,
            const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const input::schedule& schd,
            const qtensor3<Qm::ifabelian,Tm>& site,
            const qoper_dict<Qm::ifabelian,Tm>& qops1,
            const qoper_dict<Qm::ifabelian,Tm>& qops2,
            qoper_dict<Qm::ifabelian,Tm>& qops){
         int size = icomb.world.size();
         int rank = icomb.world.rank();
         const bool ifdist1 = schd.ctns.ifdist1;
         const bool ifdistc = schd.ctns.ifdistc;
         const bool ifdists = schd.ctns.ifdists;
         const int alg_renorm = schd.ctns.alg_renorm;
         const bool ifab = Qm::ifabelian;
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         const bool debug = (rank == 0); 
         if(debug and schd.ctns.verbose>0){ 
            std::cout << "ctns::oper_renorm_opS"
               << " superblock=" << superblock 
               << " ifab=" << ifab
               << " isym=" << isym 
               << " ifkr=" << ifkr
               << " ifdist1=" << ifdist1
               << " ifdists=" << ifdists 
               << " alg_renorm=" << alg_renorm	
               << " mpisize=" << size
               << std::endl;
         }
         assert(ifdist1 and ifdists);
         auto t0 = tools::get_time();

         // renormalization for opS
         if(alg_renorm == 0){

            oper_renorm_kernel_opS(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);

         }else if(alg_renorm == 1){

            symbolic_kernel_renorm_opS(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);

         }else if(alg_renorm == 4){

            preprocess_renorm_opS(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);

         }else if(alg_renorm == 6 || alg_renorm == 7 || alg_renorm == 8 || alg_renorm == 9){

            preprocess_renorm_batch_opS(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);

#ifdef GPU
         }else if(alg_renorm == 16 || alg_renorm == 17 || alg_renorm == 18 || alg_renorm == 19){

            /*
               preprocess_renorm_batchGPU_opS(superblock, icomb, int2e, schd, site, qops1, qops2, qops, size, rank);
               */

#endif
         }else{
            std::cout << "error: no such option for alg_renorm=" << alg_renorm << std::endl;
            exit(1);
         } // alg_renorm

         auto t1 = tools::get_time();
         if(debug){
            double t_tot = tools::get_duration(t1-t0);
            std::cout << "----- TIMING FOR oper_renorm_opS : " << t_tot << " S"
               << " rank=" << rank << " -----"
               << std::endl;
         }
      }

#endif // SERIAL

} // ctns

#endif
