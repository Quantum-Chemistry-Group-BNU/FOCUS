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
#include "preprocess_formulae.h"
#include "preprocess_sigma.h"
#include "preprocess_sigma2.h"
#include "preprocess_sigma_batch.h"
#ifdef GPU
#include "preprocess_sigma_batchGPU.h"
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

         std::cout << "start sweep_twodot" << std::endl;

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

         std::cout << "start rank=" << rank << " sweep_twodot" << std::endl;

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
         std::cout << "rank,diag0=" << rank << std::endl;
         auto time0 = tools::get_time();
         std::vector<double> diag(ndim, ecore/size); // constant term
         twodot_diag1(qops_dict, wf, diag.data(), size, rank, schd.ctns.ifdist1);
         auto time1 = tools::get_time();
         std::cout << "rank,diag1=" << rank << std::endl;

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
            boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
            diag = std::move(diag2);
         }
#endif 
         timing.tb = tools::get_time();
         std::cout << "rank,diag2=" << rank << std::endl;

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
#ifdef GPU
         Tm* dev_opaddr;
         Tm * dev_workspace;
#endif
         double t_kernel_ibond=0.0, t_reduction_ibond=0.0; // debug
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
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 
            HVec = bind(&ctns::symbolic_Hx<Tm,stensor4<Tm>>, _1, _2, std::cref(H_formulae),
                  std::cref(qops_dict), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));

         }else if(schd.ctns.alg_hvec == 2){ 

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
            HVec = bind(&ctns::symbolic_Hx2<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
                  std::cref(H_formulae), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(schd.ctns.alg_hvec == 3){

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
            HVec = bind(&ctns::symbolic_Hx3<Tm,stensor4<Tm>,qinfo4<Tm>>, _1, _2, 
                  std::cref(H_formulae2), std::cref(qops_dict), std::cref(ecore), 
                  std::ref(wf), std::cref(size), std::cref(rank), std::cref(info_dict), 
                  std::cref(opsize), std::cref(wfsize), std::cref(tmpsize),
                  std::ref(workspace));

         }else if(schd.ctns.alg_hvec == 4){

            // Single Hxlst	   
            // symbolic formulae + intermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 
            preprocess_formulae_Hxlist(qops_dict, oploc, H_formulae, wf, inter, 
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
                  icomb.world.barrier();
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
#endif 
               icomb.world.barrier();
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
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
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae); 
            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, inter, 
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
                  icomb.world.barrier();
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
#endif 
               icomb.world.barrier();
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
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
            if(schd.ctns.batchsize == 0){
               std::cout << "error: batchsize should be set!" << std::endl;
               exit(1);
            }
            // symbolic formulae + intermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);
            // gen MMlst & reorder
            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, inter, 
                  Hxlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);
            // generate mmtasks
            int icase = 0;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<Hxlst2.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.batchgemm, schd.ctns.batchsize,
                     blksize*2, schd.ctns.hxorder, icase);
            } // i
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
                  std::ref(Hxlst2), std::ref(mmtasks), std::ref(opaddr), std::ref(workspace),
                  std::ref(t_kernel_ibond), std::ref(t_reduction_ibond));

#ifdef GPU
         }else if(schd.ctns.alg_hvec == 7){

            // BatchGEMM on GPU
            if(schd.ctns.batchsize == 0 && std::abs(schd.ctns.batchmem) < 1.e-10){
               std::cout << "error: batchsize/batchmem should be set!" << std::endl;
               exit(1);
            }
            // symbolic formulae + intermediates + preallocation of workspace
            H_formulae = symbolic_formulae_twodot(qops_dict, int2e, size, rank, fname,
                  schd.ctns.sort_formulae, schd.ctns.ifdist1, debug_formulae);
            // gen MMlst & reorder
            preprocess_formulae_Hxlist2(qops_dict, oploc, H_formulae, wf, inter, 
                  Hxlst2, blksize, cost, rank==0 && schd.ctns.verbose>0);
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
                  icomb.world.barrier();
               }
               double cost_tot = cost;
#ifndef SERIAL
               if(size > 1) boost::mpi::reduce(icomb.world, cost, cost_tot, std::plus<double>(), 0);
#endif 
               icomb.world.barrier();
               if(rank == 0) std::cout << "total cost for Hx=" << cost_tot << std::endl;
            }

            // GPU: copy operators (qops_dict & inter)
            // 1. allocate memery on GPU
            size_t opertot = qops_dict.at("l").size()
               + qops_dict.at("r").size()
               + qops_dict.at("c1").size()
               + qops_dict.at("c2").size()
               + inter.size();
#if defined(USE_CUDA_OPERATION)
            CUDA_CHECK(cudaMalloc((void**)&dev_opaddr, opertot*sizeof(Tm)));
#else
            MAGMA_CHECK(magma_dmalloc((double**)(&dev_opaddr), opertot));
#endif

            Tm* dev_l_opaddr = dev_opaddr;
            Tm* dev_r_opaddr = dev_l_opaddr + qops_dict.at("l").size();
            Tm* dev_c1_opaddr = dev_r_opaddr + qops_dict.at("r").size();
            Tm* dev_c2_opaddr = dev_c1_opaddr + qops_dict.at("c1").size();
            Tm* dev_inter_opaddr = dev_c2_opaddr + qops_dict.at("c2").size();

            // 2. copy
#if defined(USE_CUDA_OPERATION)
            CUDA_CHECK(cudaMemcpy(dev_l_opaddr,qops_dict.at("l")._data,qops_dict.at("l").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_r_opaddr,qops_dict.at("r")._data,qops_dict.at("r").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_c1_opaddr,qops_dict.at("c1")._data,qops_dict.at("c1").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_c2_opaddr,qops_dict.at("c2")._data,qops_dict.at("c2").size()*sizeof(Tm), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dev_inter_opaddr,inter._data,inter.size()*sizeof(Tm), cudaMemcpyHostToDevice));
#else
            magma_dsetvector(qops_dict.at("l").size(),  (double*)qops_dict.at("l")._data, 1,  (double*)dev_l_opaddr,    1,  magma_queue);
            magma_dsetvector(qops_dict.at("r").size(),  (double*)qops_dict.at("r")._data, 1,  (double*)dev_r_opaddr,    1,  magma_queue);
            magma_dsetvector(qops_dict.at("c1").size(), (double*)qops_dict.at("c1")._data, 1, (double*)dev_c1_opaddr, 1,  magma_queue);
            magma_dsetvector(qops_dict.at("c2").size(), (double*)qops_dict.at("c2")._data, 1, (double*)dev_c2_opaddr, 1,  magma_queue);
            magma_dsetvector(inter.size(), (double*)inter._data, 1, (double*)dev_inter_opaddr,  1,  magma_queue);
#endif

            // 3. save pointers to opaddr
            opaddr[0]=dev_l_opaddr;
            opaddr[1]=dev_r_opaddr;
            opaddr[2]=dev_c1_opaddr;
            opaddr[3]=dev_c2_opaddr;
            opaddr[4]=dev_inter_opaddr;

            std::cout << "rank=" << rank << " qops(tot)=" << opertot 
               << ":" << tools::sizeMB<Tm>(opertot) << "MB"
               << ":" << tools::sizeGB<Tm>(opertot) << "GB"
               << std::endl;

            // determine the size of batch
            size_t batchsize = 0;
            if(schd.ctns.batchsize > 0){
               batchsize = schd.ctns.batchsize;
            }else{
               batchsize = std::ceil((schd.ctns.batchmem - tools::sizeGB<Tm>(opertot+2*ndim))*std::pow(1024,3)/(blksize*2*sizeof(Tm)));
            }
            if(batchsize <= 0){
               std::cout << "error: in sufficient memory!" << std::endl;
               exit(1);
            }
            // generate mmtasks
            int icase = 1;
            mmtasks.resize(Hxlst2.size());
            for(int i=0; i<Hxlst2.size(); i++){
               mmtasks[i].init(Hxlst2[i], schd.ctns.batchgemm, batchsize,
                     blksize*2, schd.ctns.hxorder, 1);
               if(rank==0){
                  std::cout << "rank=" << rank << " iblk=" << i 
                     << " mmtasks.totsize=" << mmtasks[i].totsize
                     << " batchsize=" << mmtasks[i].batchsize 
                     << " nbatch=" << mmtasks[i].nbatch 
                     << std::endl;
               }
            } // i

            // 4. allocate memory for Davidson: x,worktot
            worktot = 2*ndim + batchsize*blksize*2;
            if(debug && schd.ctns.verbose>0){
               std::cout << "rank=" << rank
                  << " blksize=" << blksize
                  << " batchsize=" << batchsize
                  << " opertot(GB)=" << tools::sizeGB<Tm>(opertot)
                  << " worktot(GB)=" << tools::sizeGB<Tm>(worktot)
                  << " gpu(GB)=" << tools::sizeGB<Tm>(opertot+worktot)
                  << std::endl;
            }
#if defined(USE_CUDA_OPERATION)
            CUDA_CHECK(cudaMalloc((void**)&dev_workspace, worktot*sizeof(Tm)));
#else
            MAGMA_CHECK(magma_dmalloc((double**)(&dev_workspace), worktot));
#endif

            HVec = bind(&ctns::preprocess_Hx_batchGPU<Tm>, _1, _2,
                  std::cref(scale), std::cref(size), std::cref(rank),
                  std::cref(ndim), std::cref(blksize), 
                  std::ref(Hxlst2), std::ref(mmtasks), std::ref(opaddr), std::ref(dev_workspace),
                  std::ref(t_kernel_ibond), std::ref(t_reduction_ibond));
#endif

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
            if(schd.ctns.alg_hvec != 7) delete[] workspace;
#ifdef GPU
            // free memory space on GPU
            if(schd.ctns.alg_hvec == 7){
#if defined(USE_CUDA_OPERATION)
               CUDA_CHECK(cudaFree(dev_opaddr));
               CUDA_CHECK(cudaFree(dev_workspace));
#else
               MAGMA_CHECK(magma_free(dev_opaddr));
               MAGMA_CHECK(magma_free(dev_workspace));
#endif
            }
#endif
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
         std::cout << "### lzd ###" << rank << std::endl;
         qops_pool.save(frop);
         std::cout << "=== lzd ===" << rank << std::endl;
         oper_remove(fdel, debug);
         std::cout << "--- lzd ---" << rank << std::endl;

         timing.t1 = tools::get_time();
         if(debug) timing.analysis("time_local", schd.ctns.verbose>0);
         std::cout << "end rank=" << rank << " sweep_twodot" << std::endl;
      }

} // ctns

#endif
