#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_twodot_renorm.h"
#include "sweep_twodot_diag.h"
#include "sweep_twodot_diag1.h"
#include "sweep_twodot_diag2.h"
#include "sweep_twodot_local.h"
#include "sweep_twodot_sigma.h"
#include "symbolic_formulae_twodot.h"
#include "symbolic_kernel_sigma.h"
#include "symbolic_kernel_sigma2.h"
#include "symbolic_kernel_sigma3.h"
#include "preprocess_size.h"
#include "preprocess_hxlist.h"
#include "preprocess_hformulae.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma2.h"
#include "preprocess_sigma_batch.h"
#include "preprocess_sigma_batch2.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "preprocess_sigma_batchGPU.h"
#include "preprocess_sigma_batch2GPU.h"
#endif

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
         const int alg_hvec = schd.ctns.alg_hvec;
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_twodot"
               << " alg_hvec=" << alg_hvec
               << " alg_renorm=" << schd.ctns.alg_renorm
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
         }
         auto& CPUmem = sweeps.opt_CPUmem[isweep][ibond];
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
            size_t opertot = qops_dict.at("l").size()
               + qops_dict.at("r").size()
               + qops_dict.at("c1").size()
               + qops_dict.at("c2").size();
            std::cout << " qops(tot)=" << opertot 
               << ":" << tools::sizeMB<Tm>(opertot) << "MB"
               << ":" << tools::sizeGB<Tm>(opertot) << "GB"
               << std::endl;
         }
         if(debug){
            CPUmem.comb = sizeof(Tm)*icomb.display_size(); 
            CPUmem.oper = sizeof(Tm)*qops_pool.size(); 
            CPUmem.display();
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
         size_t ndim = wf.size();
         if(debug){
            std::cout << "wf4(diml,dimr,dimc1,dimc2)=(" 
               << ql.get_dimAll() << ","
               << qr.get_dimAll() << ","
               << qc1.get_dimAll() << ","
               << qc2.get_dimAll() << ")"
               << " nnz=" << ndim << ":"
               << tools::sizeMB<Tm>(ndim) << "MB"
               << std::endl;
            wf.print("wf",schd.ctns.verbose-2);
         }
         if(ndim == 0){
            std::cout << "error: symmetry is inconsistent as ndim=0" << std::endl;
            exit(1);
         }

         /*
         for(const auto& pr : qops_dict.at("r")('C')){
            std::cout << "pr=" << pr.first << " " 
               << pr.second.size() << std::endl;
         }
         */

         // 3. Davidson solver for wf
         // 3.1 diag 
         auto time0 = tools::get_time();
         std::vector<double> diag(ndim, ecore/size); // constant term
         twodot_diag1(qops_dict, wf, diag.data(), size, rank, schd.ctns.ifdist1);
         auto time1 = tools::get_time();
         /*     
                std::vector<double> diag1(ndim, ecore/size); // constant term
                twodot_diag1(qops_dict, wf, diag1.data(), size, rank, schd.ctns.ifdist1);
                auto time2 = tools::get_time();

                std::vector<double> diag2(ndim, ecore/size); // constant term
                twodot_diag2(qops_dict, wf, diag2.data(), size, rank, schd.ctns.ifdist1);
                auto time3 = tools::get_time();

                linalg::xaxpy(ndim, -1.0, diag.data(), diag1.data());
                linalg::xaxpy(ndim, -1.0, diag.data(), diag2.data());
                std::cout << "-----------lzd-----------" << std::endl;
                std::cout << "t0,t1,t2=" 
                << tools::get_duration(time1-time0) << "," 
                << tools::get_duration(time2-time1) << "," 
                << tools::get_duration(time3-time2) << "," 
                << std::endl;
                double diff1 = linalg::xnrm2(ndim, diag1.data());
                double diff2 = linalg::xnrm2(ndim, diag1.data());
                std::cout << "diff of diag1,diag2=" << diff1 << "," << diff2 << std::endl;
                std::cout << "-----------lzd-----------" << std::endl;
                if(diff1 > 1.e-8 || diff2 > 1.e-8){ 
                std::cout << "diff is too large!" << std::endl;
                exit(1);
                }
                */
#ifndef SERIAL
         // reduction of partial diag: no need to broadcast, if only rank=0 
         // executes the preconditioning in Davidson's algorithm
         if(size > 1){
            std::vector<double> diag2(ndim);
            mpi_wrapper::reduce(icomb.world, diag.data(), ndim, diag2.data(), std::plus<double>(), 0);
            diag = std::move(diag2);
         }
#endif 
         timing.tb = tools::get_time();

         // 3.2 Solve local problem: Hc=cE
         // prepare HVec
         std::map<qsym,qinfo4<Tm>> info_dict;
         size_t opsize=0, wfsize=0, tmpsize=0, worktot=0;
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
         hintermediates<Tm> hinter; // hvec4,5,6
         Hxlist<Tm> Hxlst; // hvec4
         Hxlist2<Tm> Hxlst2; // hvec5
         MMtasks<Tm> mmtasks; // hvec6
         Tm scale = Km::ifkr? 0.5*ecore : 1.0*ecore;
         std::map<std::string,int> oploc = {{"l",0},{"r",1},{"c1",2},{"c2",3}};
         Tm* opaddr[5] = {qops_dict.at("l")._data, qops_dict.at("r")._data,
            qops_dict.at("c1")._data, qops_dict.at("c2")._data,
            nullptr};
         size_t blksize=0, blksize0=0;
         double cost=0.0;
         Tm* workspace;
#ifdef GPU
         Tm* dev_opaddr[5] = {nullptr,nullptr,nullptr,nullptr,nullptr};
         Tm* dev_oper = nullptr;
         Tm* dev_workspace = nullptr;
         size_t GPUmem_used = 0;
#endif
         double t_kernel_ibond=0.0, t_reduction_ibond=0.0; // debug
         using std::placeholders::_1;
         using std::placeholders::_2;
         const bool debug_formulae = schd.ctns.verbose>0;
         if(Km::ifkr && alg_hvec >=4){
            std::cout << "error: alg_hvec >= 4 does not support complex yet!" << std::endl;
            exit(1); 
         }
         if(alg_hvec == 0){

            // oldest version
            Hx_funs = twodot_Hx_functors(qops_dict, int2e, ecore, wf, size, rank, 
                  schd.ctns.ifdist1, debug_formulae);
            HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 1){

            // raw version: symbolic formulae + dynamic allocation of memory 
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 
            HVec = bind(&ctns::symbolic_Hx<Tm,stensor4<Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(alg_hvec == 2){ 

            // symbolic formulae + preallocation of workspace 
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);
            tmpsize = opsize + 3*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot;
            HVec = bind(&ctns::symbolic_Hx2<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 3){

            // symbolic formulae (factorized) + preallocation of workspace 
            H_formulae2 = symbolic_formulae_twodot2(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 
            tmpsize = opsize + 4*wfsize;
            worktot = maxthreads*tmpsize;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: opsize=" << opsize << " wfsize=" << wfsize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot;
            HVec = bind(&ctns::symbolic_Hx3<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(alg_hvec == 4){

            // Single Hxlst: symbolic formulae + hintermediates + preallocation of workspace

            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);

            hinter.init(schd.ctns.alg_hinter, qops_dict, oploc, opaddr, H_formulae, debug);
            if(debug) CPUmem.oper += sizeof(Tm)*hinter.size();

            preprocess_formulae_Hxlist(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst, blksize, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist(Hxlst, schd.ctns.hxorder);

            // debug hxlst
            if(schd.ctns.verbose>0){
               for(int k=0; k<size; k++){
                  if(rank == k){
                     if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
                     std::cout << " * rank=" << k 
                        << " size(H_formulae)=" << H_formulae.size() 
                        << " size(Hxlst)=" << Hxlst.size()
                        << " blksize=" << blksize
                        << " cost=" << cost 
                        << std::endl;
                  }
#ifndef SERIAL
                  icomb.world.barrier();
#endif
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
               icomb.world.barrier();
#endif 
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
            }

            worktot = maxthreads*(blksize*2+ndim);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot; // private workspace for each thread allocated inside preprocess_Hx

            HVec = bind(&ctns::preprocess_Hx<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst), std::ref(opaddr));

         }else if(alg_hvec == 5){

            // Hxlist2: symbolic formulae + hintermediates + preallocation of workspace

            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 

            hinter.init(schd.ctns.alg_hinter, qops_dict, oploc, opaddr, H_formulae, debug);
            if(debug) CPUmem.oper += sizeof(Tm)*hinter.size();

            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);

            get_MMlist(Hxlst2, schd.ctns.hxorder);

            // debug hxlst
            if(schd.ctns.verbose>0){
               for(int k=0; k<size; k++){
                  if(rank == k){
                     if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
                     size_t hxsize = 0;
                     for(int i=0; i<Hxlst2.size(); i++){
                        hxsize += Hxlst2[i].size();
                     }
                     std::cout << " * rank=" << k
                        << " size(H_formulae)=" << H_formulae.size() 
                        << " size(Hxlst)=" << hxsize 
                        << " blksize=" << blksize
                        << " cost=" << cost
                        << std::endl;
                  }
#ifndef SERIAL
                  icomb.world.barrier();
#endif
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
               icomb.world.barrier();
#endif 
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
            }

            worktot = maxthreads*blksize*3;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot; // private workspace for each thread

            HVec = bind(&ctns::preprocess_Hx2<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst2), std::ref(opaddr));

         }else if(alg_hvec == 6){

            if(schd.ctns.alg_hinter != 0 and schd.ctns.alg_hinter != 1){
               std::cout << "error: alg_hvec=6 should be used with alg_hinter=0 or 1!" << std::endl;
               exit(1);
            }
            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }

            // BatchGEMM: symbolic formulae + hintermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);

            hinter.init(schd.ctns.alg_hinter, qops_dict, oploc, opaddr, H_formulae, debug);
            if(debug) CPUmem.oper += sizeof(Tm)*hinter.size();

            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Hxlst2.size(); i++){
               maxbatch = std::max(maxbatch, Hxlst2[i].size());
            } // i
            size_t batchsize = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;
            
            // generate mmtasks
            const int batchblas = 1;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<mmtasks.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2);
               if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i 
                     << " size=" << Hxlst2[i][0].size 
                     << " mmtasks.totsize=" << mmtasks[i].totsize
                     << " batchsize=" << mmtasks[i].batchsize 
                     << " nbatch=" << mmtasks[i].nbatch 
                     << std::endl;
               }
            } // i

            worktot = batchsize*blksize*2;
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot;

            HVec = bind(&ctns::preprocess_Hx_batch<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::ref(mmtasks), std::ref(opaddr), std::ref(workspace),
                  std::ref(t_kernel_ibond), std::ref(t_reduction_ibond));

         }else if(alg_hvec == 7){

            if(schd.ctns.alg_hinter != 1){
               std::cout << "error: alg_hvec=7 should be used with alg_hinter=1!" << std::endl;
               exit(1);
            }
            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }

            // BatchGEMM: symbolic formulae + hintermediates [on the fly] + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);

            hinter.initDirect(schd.ctns.alg_hinter, qops_dict, oploc, opaddr, H_formulae, debug);
            if(debug) CPUmem.oper += sizeof(Tm)*hinter.size();

            preprocess_formulae_Hxlist2Direct(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Hxlst2.size(); i++){
               maxbatch = std::max(maxbatch, Hxlst2[i].size());
            } // i
            size_t batchsize = (maxbatch < schd.ctns.batchsize)? maxbatch : schd.ctns.batchsize;
 
            // generate mmtasks
            const int batchblas = 1;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<mmtasks.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i 
                     << " size=" << Hxlst2[i][0].size 
                     << " mmtasks.totsize=" << mmtasks[i].totsize
                     << " batchsize=" << mmtasks[i].batchsize 
                     << " nbatch=" << mmtasks[i].nbatch 
                     << std::endl;
               }
            } // i

            worktot = batchsize*(blksize*2+blksize0);
            if(debug && schd.ctns.verbose>0){
               std::cout << "preprocess for Hx: ndim=" << ndim << " blksize=" << blksize 
                  << " blksize0=" << blksize0 << " batchsize=" << batchsize
                  << " worktot=" << worktot << ":" << tools::sizeMB<Tm>(worktot) << "MB"
                  << ":" << tools::sizeGB<Tm>(worktot) << "GB" << std::endl; 
            }
            workspace = new Tm[worktot];
            if(debug) CPUmem.hvec = sizeof(Tm)*worktot;

            opaddr[4] = workspace + batchsize*(blksize*2); // memory layout [workspace|inter]
            HVec = bind(&ctns::preprocess_Hx_batch2<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::ref(mmtasks), std::ref(opaddr), std::ref(workspace),
                  std::ref(hinter._data));

#ifdef GPU
         }else if(alg_hvec == 16){

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

            // BatchGEMM on GPU: symbolic formulae + hintermediates + preallocation of workspace
            timing.tb1 = tools::get_time();

            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);

            timing.tb2 = tools::get_time();

            // allocate memery on GPU & copy qops
            size_t opertot = qops_dict.at("l").size()
               + qops_dict.at("r").size()
               + qops_dict.at("c1").size()
               + qops_dict.at("c2").size();
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
            dev_opaddr[1] = dev_opaddr[0] + qops_dict.at("l").size();
            dev_opaddr[2] = dev_opaddr[1] + qops_dict.at("r").size();
            dev_opaddr[3] = dev_opaddr[2] + qops_dict.at("c1").size();
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_opaddr[0], qops_dict.at("l")._data, qops_dict.at("l").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[1], qops_dict.at("r")._data, qops_dict.at("r").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[2], qops_dict.at("c1")._data, qops_dict.at("c1").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[3], qops_dict.at("c2")._data, qops_dict.at("c2").size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_opaddr[0], qops_dict.at("l")._data, qops_dict.at("l").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[1], qops_dict.at("r")._data, qops_dict.at("r").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[2], qops_dict.at("c1")._data, qops_dict.at("c1").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[3], qops_dict.at("c2")._data, qops_dict.at("c2").size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            timing.tb3 = tools::get_time();

            //------------------------------
            // intermediates
            //------------------------------
            if(schd.ctns.alg_hinter != 2){
               // compute hintermediates on CPU
               hinter.init(schd.ctns.alg_hinter, qops_dict, oploc, opaddr, H_formulae, debug);
            }else{
               // compute hintermediates on GPU directly
               hinter.init(schd.ctns.alg_hinter, qops_dict, oploc, dev_opaddr, H_formulae, debug);
            }
            timing.tb4 = tools::get_time();
            
            size_t GPUmem_hinter = sizeof(Tm)*hinter.size();
            // copy from CPU to GPU 
            if(schd.ctns.alg_hinter != 2){
               if(debug) CPUmem.oper += GPUmem_hinter;
               dev_opaddr[4] = (Tm*)GPUmem.allocate(GPUmem_hinter);
#ifdef USE_HIP
               HIP_CHECK(hipMemcpy(dev_opaddr[4], hinter._value.data(), GPUmem_hinter, hipMemcpyHostToDevice));
#else
               CUDA_CHECK(cudaMemcpy(dev_opaddr[4], hinter._value.data(), GPUmem_hinter, cudaMemcpyHostToDevice));
#endif// USE_HIP
            }
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter)(GB)=" << GPUmem_oper/std::pow(1024.0,3)
                  << "," << GPUmem_hinter/std::pow(1024.0,3) 
                  << std::endl;
            }
            timing.tb5 = tools::get_time();

            // gen MMlst & reorder
            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst2, blksize, cost, debug && schd.ctns.verbose>0);
            if(schd.ctns.verbose>0){
               // debug hxlst
               for(int k=0; k<size; k++){
                  if(rank == k){
                     if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
                     size_t hxsize = 0;
                     for(int i=0; i<Hxlst2.size(); i++){
                        hxsize += Hxlst2[i].size();
                     }
                     std::cout << " * rank=" << k
                        << " size(H_formulae)=" << H_formulae.size() 
                        << " size(Hxlst)=" << hxsize 
                        << " blksize=" << blksize
                        << " cost=" << cost
                        << std::endl;
                  }
#ifndef SERIAL
                  icomb.world.barrier();
#endif
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
               icomb.world.barrier();
#endif 
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
            }
            timing.tb6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Hxlst2.size(); i++){
               maxbatch = std::max(maxbatch, Hxlst2[i].size());
            } // i
            //
            // Determine batchsize dynamically: total GPU memory required
            //   oper + dvdson + N*blksize*2 
            // + [6*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*)] (inside BatchGEMM)
            // + N*sizeof(double/complex*) [inside GEMV]
            //
            // 6*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*) estimated by 6*8*(N+1) + 3*N*16 = 96*N+48
            // N*sizeof(double/complex*) estimated by N*16 = 16*N
            //
            size_t batchsize = 0;
            size_t GPUmem_dvdson = sizeof(Tm)*2*ndim;
            size_t GPUmem_reserved = GPUmem_oper + GPUmem_hinter + GPUmem_dvdson + 48;
            if(GPUmem.size() > GPUmem_reserved){
               batchsize = std::floor(double(GPUmem.size() - GPUmem_reserved)/(sizeof(Tm)*blksize*2 + 112));
               batchsize = (maxbatch < batchsize)? maxbatch : batchsize; // sufficient
               if(batchsize == 0 && maxbatch != 0){
                  std::cout << "error: in sufficient GPU memory: batchsize=0!" << std::endl;
                  exit(1);
               }
            }else{
               std::cout << "error: in sufficient GPU memory for batchGEMM! already reserved:" << std::endl;
               std::cout << "GPUmem.size=" << GPUmem.size() << " GPUmem.used=" << GPUmem.used()
                  << " GPUmem_reserved=" << GPUmem_reserved << " (oper,hinter,dvdson)=" 
                  << GPUmem_oper << "," << GPUmem_hinter << "," << GPUmem_dvdson 
                  << std::endl;
               exit(1);
            }
            size_t GPUmem_batch = sizeof(Tm)*batchsize*blksize*2;
            dev_workspace = (Tm*)GPUmem.allocate(GPUmem_dvdson + GPUmem_batch);
            GPUmem_used = GPUmem.used(); // oper + dvdson + batch, later used in deallocate
            if(schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter,dvdson,batch)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_hinter/std::pow(1024.0,3)
                  << "," << GPUmem_dvdson/std::pow(1024.0,3)
                  << "," << GPUmem_batch/std::pow(1024.0,3)
                  << " batchsize=" << batchsize
                  << std::endl;
            }

            // generate mmtasks given batchsize
            const int batchblas = 2;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<mmtasks.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2);
               if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i 
                     << " size=" << Hxlst2[i][0].size 
                     << " mmtasks.totsize=" << mmtasks[i].totsize
                     << " batchsize=" << mmtasks[i].batchsize 
                     << " nbatch=" << mmtasks[i].nbatch 
                     << std::endl;
               }
            } // i
            // save for analysis of BatchGEMM
            if(debug && isweep == schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond){
               save_mmtasks(mmtasks, isweep, ibond);
            }
            timing.tb7 = tools::get_time();

            // GPU version of Hx
            HVec = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::ref(mmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                  std::ref(t_kernel_ibond), std::ref(t_reduction_ibond));

         }else if(alg_hvec == 17){

            if(schd.ctns.alg_hinter != 2){
               std::cout << "error: alg_hvec=17 must be used with alg_hinter=2!" << std::endl;
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

            // BatchGEMM on GPU: symbolic formulae + hintermediates + preallocation of workspace
            timing.tb1 = tools::get_time();

            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);

            timing.tb2 = tools::get_time();

            // allocate memery on GPU & copy qops
            size_t opertot = qops_dict.at("l").size()
               + qops_dict.at("r").size()
               + qops_dict.at("c1").size()
               + qops_dict.at("c2").size();
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
            dev_opaddr[1] = dev_opaddr[0] + qops_dict.at("l").size();
            dev_opaddr[2] = dev_opaddr[1] + qops_dict.at("r").size();
            dev_opaddr[3] = dev_opaddr[2] + qops_dict.at("c1").size();
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_opaddr[0], qops_dict.at("l")._data, qops_dict.at("l").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[1], qops_dict.at("r")._data, qops_dict.at("r").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[2], qops_dict.at("c1")._data, qops_dict.at("c1").size()*sizeof(Tm), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(dev_opaddr[3], qops_dict.at("c2")._data, qops_dict.at("c2").size()*sizeof(Tm), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_opaddr[0], qops_dict.at("l")._data, qops_dict.at("l").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[1], qops_dict.at("r")._data, qops_dict.at("r").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[2], qops_dict.at("c1")._data, qops_dict.at("c1").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_opaddr[3], qops_dict.at("c2")._data, qops_dict.at("c2").size()*sizeof(Tm), cudaMemcpyHostToDevice));
#endif //USE_HIP

            timing.tb3 = tools::get_time();
            
            hinter.initDirect(schd.ctns.alg_hinter, qops_dict, oploc, dev_opaddr, H_formulae, debug);

            timing.tb4 = tools::get_time();

            size_t GPUmem_hinter = sizeof(Tm)*hinter.size();
            if(rank == 0 && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter)(GB)=" << GPUmem_oper/std::pow(1024.0,3)
                  << "," << GPUmem_hinter/std::pow(1024.0,3) 
                  << std::endl;
            }
 
            timing.tb5 = tools::get_time();

            // gen MMlst & reorder
            preprocess_formulae_Hxlist2Direct(qops_dict, oploc, H_formulae, wf, hinter, 
                  Hxlst2, blksize, blksize0, cost, rank==0 && schd.ctns.verbose>0);
            if(schd.ctns.verbose>0){
               // debug hxlst
               for(int k=0; k<size; k++){
                  if(rank == k){
                     if(rank == 0) std::cout << "partition of Hxlst:" << std::endl;
                     size_t hxsize = 0;
                     for(int i=0; i<Hxlst2.size(); i++){
                        hxsize += Hxlst2[i].size();
                     }
                     std::cout << " * rank=" << k
                        << " size(H_formulae)=" << H_formulae.size() 
                        << " size(Hxlst)=" << hxsize 
                        << " blksize=" << blksize
                        << " cost=" << cost
                        << std::endl;
                  }
#ifndef SERIAL
                  icomb.world.barrier();
#endif
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
               icomb.world.barrier();
#endif 
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
            }
            timing.tb6 = tools::get_time();

            // compute batchsize & allocate workspace
            size_t maxbatch = 0;
            for(int i=0; i<Hxlst2.size(); i++){
               maxbatch = std::max(maxbatch, Hxlst2[i].size());
            } // i
            
            //
            // Determine batchsize dynamically: total GPU memory required
            //   oper + dvdson + N*blksize*2 
            // + [6*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*)] (inside BatchGEMM)
            // + N*sizeof(double/complex*) [inside GEMV]
            // + [5*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*)] (inside BatchGEMV)
            //
            // 6*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*) estimated by 6*8*(N+1) + 3*N*16 = 96*N+48
            // N*sizeof(double/complex*) estimated by N*16 = 16*N
            // 5*(N+1)*sizeof(int) + 3*N*sizeof(double/complex*) estimated by 5*8*(N+1) + 3*N*16 = 88*N+40
            //
            size_t batchsize = 0;
            size_t GPUmem_dvdson = sizeof(Tm)*2*ndim;
            size_t GPUmem_reserved = GPUmem_oper + GPUmem_hinter + GPUmem_dvdson + 88;
            if(GPUmem.size() > GPUmem_reserved){
               batchsize = std::floor(double(GPUmem.size() - GPUmem_reserved)/(sizeof(Tm)*(blksize*2+blksize0) + 200));
               batchsize = (maxbatch < batchsize)? maxbatch : batchsize; // sufficient
               if(batchsize == 0 && maxbatch != 0){
                  std::cout << "error: in sufficient GPU memory: batchsize=0!" << std::endl;
                  exit(1);
               }
            }else{
               std::cout << "error: in sufficient GPU memory for batchGEMM! already reserved:" << std::endl;
               std::cout << "GPUmem.size=" << GPUmem.size() << " GPUmem.used=" << GPUmem.used()
                  << " GPUmem_reserved=" << GPUmem_reserved << " (oper,hinter,dvdson)=" 
                  << GPUmem_oper << "," << GPUmem_hinter << "," << GPUmem_dvdson 
                  << std::endl;
               exit(1);
            }
            size_t GPUmem_batch = sizeof(Tm)*batchsize*(blksize*2+blksize0);
            dev_workspace = (Tm*)GPUmem.allocate(GPUmem_dvdson + GPUmem_batch);
            GPUmem_used = GPUmem.used(); // oper + dvdson + batch, later used in deallocate
            if(schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " GPUmem.size(GB)=" << GPUmem.size()/std::pow(1024.0,3)
                  << " GPUmem.used(GB)=" << GPUmem.used()/std::pow(1024.0,3)
                  << " (oper,hinter,dvdson,batch)(GB)=" << GPUmem_oper/std::pow(1024.0,3) 
                  << "," << GPUmem_hinter/std::pow(1024.0,3)
                  << "," << GPUmem_dvdson/std::pow(1024.0,3)
                  << "," << GPUmem_batch/std::pow(1024.0,3)
                  << " batchsize=" << batchsize
                  << std::endl;
            }

            // generate mmtasks given batchsize
            const int batchblas = 2;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<mmtasks.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.hxorder, batchblas, 
                     batchsize, blksize*2, blksize0);
               if(debug && schd.ctns.verbose>1 && Hxlst2[i].size()>0){
                  std::cout << " rank=" << rank << " iblk=" << i 
                     << " size=" << Hxlst2[i][0].size 
                     << " mmtasks.totsize=" << mmtasks[i].totsize
                     << " batchsize=" << mmtasks[i].batchsize 
                     << " nbatch=" << mmtasks[i].nbatch 
                     << std::endl;
               }
            } // i
            // save for analysis of BatchGEMM
            if(debug && isweep == schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond){
               save_mmtasks(mmtasks, isweep, ibond);
            }
            timing.tb7 = tools::get_time();

            dev_opaddr[4] = dev_workspace + 2*ndim + batchsize*(blksize*2); // memory layout [workspace|inter]
            // GPU version of Hx
            HVec = bind(&ctns::preprocess_Hx_batch2GPU<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::ref(mmtasks), std::ref(dev_opaddr), std::ref(dev_workspace),
                  std::ref(hinter._data),
                  std::ref(t_kernel_ibond), std::ref(t_reduction_ibond));
#endif // GPU

         }else{
            std::cout << "error: no such option for alg_hvec=" << alg_hvec << std::endl;
            exit(1);
         } // alg_hvec

         //-------------
         // solve HC=CE         
         //-------------
         int neig = sweeps.nroots;
         linalg::matrix<Tm> vsol(ndim,neig);
         if(debug){
            CPUmem.dvdson += sizeof(Tm)*ndim*(neig + std::min(ndim,size_t(neig+schd.ctns.nbuff))*3); 
            CPUmem.display();
         }
         auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
         auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
         oper_timer.start();
         twodot_localCI(icomb, schd, sweeps.ctrls[isweep].eps, (schd.nelec)%2,
               ndim, neig, diag, HVec, eopt, vsol, nmvp, wf, dbond);
         if(debug && schd.ctns.verbose>0){
            sweeps.print_eopt(isweep, ibond);
            if(alg_hvec == 0) oper_timer.analysis();
         }
         timing.tb8 = tools::get_time();
         sweeps.t_kernel_total[isweep] += t_kernel_ibond;
         sweeps.t_reduction_total[isweep] += t_reduction_ibond;

         // free tmp space on CPU
         if(alg_hvec==2 || alg_hvec==3 || 
               alg_hvec==6 || alg_hvec==7){
            delete[] workspace;
            if(debug) CPUmem.hvec = 0;
         }
#ifdef GPU
         if(alg_hvec>10){
            GPUmem.deallocate(dev_oper, GPUmem_used);
         }
#endif
         timing.tc = tools::get_time();

         // 3. decimation & renormalize operators
         auto fbond = icomb.topo.get_fbond(dbond, scratch, debug && schd.ctns.verbose>0);
         auto frop = fbond.first;
         auto fdel = fbond.second;
         twodot_renorm(icomb, int2e, int1e, schd, scratch, 
               vsol, wf, qops_dict, qops_pool(frop), 
               sweeps, isweep, ibond);
         if(debug){
            CPUmem.comb = sizeof(Tm)*icomb.display_size();
            CPUmem.oper = sizeof(Tm)*qops_pool.size();
            CPUmem.display();
         }
         timing.tf = tools::get_time();

         // 4. save on disk 
         qops_pool.save(frop);
         /* NOTE: At the boundary case [ -*=>=*-* and -*=<=*-* ],
            removing in the later configuration must wait until 
            the file from the former configuration has been saved!
            Therefore, oper_remove must come later than save,
            which contains the synchronization!
            */
         oper_remove(fdel, debug);
         // save for restart
         if(rank == 0){
            // local result
            std::string fresult = scratch+"/result_ibond"+std::to_string(ibond)+".info";
            rcanon_save(sweeps.opt_result[isweep][ibond], fresult);
            // updated site
            std::string fsite = scratch+"/site_ibond"+std::to_string(ibond)+".info";
            const auto p = dbond.get_current();
            const auto& pdx = icomb.topo.rindex.at(p); 
            rcanon_save(icomb.sites[pdx], fsite);
            // generated cpsi
            if(schd.ctns.guess){ 
               std::string fcpsi = scratch+"/cpsi_ibond"+std::to_string(ibond)+".info";
               rcanon_save(icomb.cpsi, fcpsi);
            }
         } // only rank-0 save and load, later broadcast

         timing.t1 = tools::get_time();
         if(debug) timing.analysis("time_local", schd.ctns.verbose>0);
      }

} // ctns

#endif
