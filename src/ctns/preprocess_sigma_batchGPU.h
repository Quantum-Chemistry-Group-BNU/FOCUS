#ifdef GPU

#ifndef PREPROCESS_SIGMA_BATCHGPU_H
#define PREPROCESS_SIGMA_BATCHGPU_H

#include "preprocess_hinter.h"
#include "preprocess_hmu.h"
#include "preprocess_hmmtask.h"
#include "../gpu/gpu_env.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   template <typename Tm>
      void analyze_tcommgpu(const size_t ndim, 
            const double time_comm1, 
            const double time_comm2){
         double sizeGB = tools::sizeGB<Tm>(ndim);
         std::cout << " ndim=" << ndim << ":" << sizeGB << "GB" 
            << " t[comm(intra)]=" << time_comm1
            << " speed=" << 2*sizeGB/time_comm1 << "GB/S" 
            << " t[comm(inter)]=" << time_comm2
            << " speed=" << 2*sizeGB/time_comm2 << "GB/S"
            << std::endl;
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchGPU(Tm* yCPU,
            const Tm* xCPU,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtasks<Tm>& Hmmtasks,
            Tm** opaddr,
            Tm* dev_workspace,
            Tm* dev_red,
            const bool ifnccl){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batchGPU"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         auto t0 = tools::get_time();
         double time_comm1=0.0, time_comm2=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         // initialization
         Tm* xGPU = dev_workspace;
         Tm* yGPU = dev_workspace + ndim;
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = xGPU;
         ptrs[6] = dev_workspace + 2*ndim;

         GPUmem.memset(yGPU, ndim*sizeof(Tm));

         // from xCPU to x &  memset yGPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
            // bcast
            auto t0comm2 = tools::get_time();
            nccl_comm.broadcast(xGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
#endif         
         }

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // gemm on GPU
               auto t0gemm = tools::get_time();
               Hmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
               auto t1reduction = tools::get_time();
               // timing
               time_gemm += tools::get_duration(t1gemm-t0gemm); 
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         // copy yGPU to yCPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            // reduce
            auto t0comm2 = tools::get_time();
            nccl_comm.reduce(yGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
            // tocpu
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#endif
         }

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            auto t1 = tools::get_time();
            std::cout << "preprocess_Hx_batchGPU: t[comm,gemm,reduction,tot]="
               << time_comm1+time_comm2 << "," << time_gemm << "," << time_reduction
               << "," << tools::get_duration(t1-t0) 
               << std::endl;
            analyze_tcommgpu<Tm>(ndim,time_comm1,time_comm2); 
            oper_timer.tcommgpu += time_comm1+time_comm2;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchDirectGPU(Tm* yCPU,
            const Tm* xCPU,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtasks<Tm>& Hmmtasks,
            Tm** opaddr,
            Tm* dev_workspace,
            Tm* alphas,
            Tm* dev_red,
            const bool ifnccl){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batchDirectGPU"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         auto t0 = tools::get_time();
         double time_comm1=0.0, time_comm2=0.0;
         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         // initialization
         Tm* xGPU = dev_workspace;
         Tm* yGPU = dev_workspace + ndim;
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = xGPU;
         ptrs[6] = dev_workspace + 2*ndim;

         GPUmem.memset(yGPU, ndim*sizeof(Tm));

         // from xCPU to x &  memset yGPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
            // bcast
            auto t0comm2 = tools::get_time();
            nccl_comm.broadcast(xGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
#endif         
         }

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // inter
               auto t0inter = tools::get_time();
               Hmmtask.inter(k, opaddr, alphas);
               auto t1inter = tools::get_time();
               // gemm on GPU
               auto t0gemm = tools::get_time();
               Hmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
               auto t1reduction = tools::get_time();
               // timing
               time_inter += tools::get_duration(t1inter-t0inter); 
               time_gemm += tools::get_duration(t1gemm-t0gemm); 
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         // copy yGPU to yCPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            // reduce
            auto t0comm2 = tools::get_time();
            nccl_comm.reduce(yGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
            // tocpu
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#endif
         }

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            auto t1 = tools::get_time();
            std::cout << "preprocess_Hx_batchDirectGPU: T(comm,inter,gemm,reduction,tot)="
               << time_comm1+time_comm2 << "," << time_inter << "," << time_gemm << "," << time_reduction
               << "," << tools::get_duration(t1-t0)
               << std::endl;
            analyze_tcommgpu<Tm>(ndim,time_comm1,time_comm2);
            oper_timer.tcommgpu += time_comm1+time_comm2;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchGPUSingle(Tm* yCPU,
            const Tm* xCPU,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtask<Tm>& Hmmtask,
            Tm** opaddr,
            Tm* dev_workspace,
            Tm* dev_red,
            const bool ifnccl){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batchGPUSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         auto t0 = tools::get_time();
         double time_comm1=0.0, time_comm2=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         // initialization
         Tm* xGPU = dev_workspace;
         Tm* yGPU = dev_workspace + ndim;
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = xGPU;
         ptrs[6] = dev_workspace + 2*ndim;

         GPUmem.memset(yGPU, ndim*sizeof(Tm));

         // from xCPU to x &  memset yGPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
            // bcast
            auto t0comm2 = tools::get_time();
            nccl_comm.broadcast(xGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
#endif         
         }

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // gemm on GPU
            auto t0gemm = tools::get_time();
            Hmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
            auto t1reduction = tools::get_time();
            // timing
            time_gemm += tools::get_duration(t1gemm-t0gemm); 
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         // copy yGPU to yCPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            // reduce
            auto t0comm2 = tools::get_time();
            nccl_comm.reduce(yGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
            // tocpu
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#endif
         }

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            auto t1 = tools::get_time();
            std::cout << "preprocess_Hx_batchGPUSingle: t[comm,gemm,reduction,tot]="
               << time_comm1+time_comm2 << "," << time_gemm << "," << time_reduction 
               << "," << tools::get_duration(t1-t0)
               << std::endl;
            analyze_tcommgpu<Tm>(ndim,time_comm1,time_comm2);
            oper_timer.tcommgpu += time_comm1+time_comm2;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchDirectGPUSingle(Tm* yCPU,
            const Tm* xCPU,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtask<Tm>& Hmmtask,
            Tm** opaddr,
            Tm* dev_workspace,
            Tm* alphas,
            Tm* dev_red,
            const bool ifnccl){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batchDirectGPUSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }
         auto t0 = tools::get_time();
         double time_comm1=0.0, time_comm2=0.0;
         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         // initialization
         Tm* xGPU = dev_workspace;
         Tm* yGPU = dev_workspace + ndim;
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = xGPU;
         ptrs[6] = dev_workspace + 2*ndim;

         GPUmem.memset(yGPU, ndim*sizeof(Tm));

         // from xCPU to x &  memset yGPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
            // bcast
            auto t0comm2 = tools::get_time();
            nccl_comm.broadcast(xGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
#endif         
         }

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // inter
            auto t0inter = tools::get_time();
            Hmmtask.inter(k, opaddr, alphas);
            auto t1inter = tools::get_time();
            // gemm on GPU
            auto t0gemm = tools::get_time();
            Hmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
            auto t1reduction = tools::get_time();
            // timing
            time_inter += tools::get_duration(t1inter-t0inter); 
            time_gemm += tools::get_duration(t1gemm-t0gemm); 
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         // copy yGPU to yCPU
         if(!ifnccl){
            auto t0comm1 = tools::get_time();
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#ifdef NCCL
         }else{
            // reduce
            auto t0comm2 = tools::get_time();
            nccl_comm.reduce(yGPU, ndim, 0);
            auto t1comm2 = tools::get_time();
            time_comm2 += tools::get_duration(t1comm2-t0comm2);
            // tocpu
            auto t0comm1 = tools::get_time();
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
            auto t1comm1 = tools::get_time();
            time_comm1 += tools::get_duration(t1comm1-t0comm1);
#endif
         }

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            auto t1 = tools::get_time();
            std::cout << "preprocess_Hx_batchDirectGPUSingle: T(comm,inter,gemm,reduction,tot)="
               << time_comm1+time_comm2 << "," << time_inter << "," << time_gemm << "," << time_reduction
               << "," << tools::get_duration(t1-t0)
               << std::endl;
            analyze_tcommgpu<Tm>(ndim,time_comm1,time_comm2);
            oper_timer.tcommgpu += time_comm1+time_comm2;
            oper_timer.sigma.analysis();
         }
      }

} // ctns

#endif

#endif
