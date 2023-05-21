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

         double time_copy=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_copy, t1_copy;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

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
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            nccl_comm.broadcast(xGPU, ndim, 0);
#endif         
         }
         gettimeofday(&t1_copy, NULL);
         time_copy = ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // gemm on GPU
               gettimeofday(&t0_gemm, NULL);
               Hmmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
            } // k
         } // i

         // copy yGPU to yCPU
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            nccl_comm.reduce(yGPU, ndim, 0);
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#endif
         }
         gettimeofday(&t1_copy, NULL);
         time_copy += ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "preprocess_Hx_batchGPU: t[copy,gemm,reduction]="
                      << time_copy << "," << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
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

         double time_copy=0.0;
         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_copy, t1_copy;
         struct timeval t0_inter, t1_inter;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

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
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            nccl_comm.broadcast(xGPU, ndim, 0);
#endif         
         }
         gettimeofday(&t1_copy, NULL);
         time_copy = ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // axpy
               gettimeofday(&t0_inter, NULL);
               Hmmtask.inter(k, opaddr, alphas);
               gettimeofday(&t1_inter, NULL);
               // gemm on GPU
               gettimeofday(&t0_gemm, NULL);
               Hmmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_inter += ((double)(t1_inter.tv_sec - t0_inter.tv_sec) 
                     + (double)(t1_inter.tv_usec - t0_inter.tv_usec)/1000000.0);
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
            } // k
         } // i

         // copy yGPU to yCPU
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            nccl_comm.reduce(yGPU, ndim, 0);
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#endif
         }
         gettimeofday(&t1_copy, NULL);
         time_copy += ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "preprocess_Hx_batchDirectGPU: t[copy,inter,gemm,reduction]="
                      << time_copy << "," << time_inter << "," << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
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

         double time_copy=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_copy, t1_copy;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

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
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            nccl_comm.broadcast(xGPU, ndim, 0);
#endif         
         }
         gettimeofday(&t1_copy, NULL);
         time_copy = ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // gemm on GPU
            gettimeofday(&t0_gemm, NULL);
            Hmmtask.kernel(k, ptrs);
            gettimeofday(&t1_gemm, NULL);
            // reduction
            gettimeofday(&t0_reduction, NULL);
            Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
            gettimeofday(&t1_reduction, NULL);
            // timing
            time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                  + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
            time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                  + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
         } // k

         // copy yGPU to yCPU
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            nccl_comm.reduce(yGPU, ndim, 0);
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#endif
         }
         gettimeofday(&t1_copy, NULL);
         time_copy += ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "preprocess_Hx_batchGPUSingle: t[copy,gemm,reduction]="
                      << time_copy << "," << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
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

         double time_copy=0.0;
         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_copy, t1_copy;
         struct timeval t0_inter, t1_inter;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

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
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            if(rank==0) GPUmem.to_gpu(xGPU, xCPU, ndim*sizeof(Tm));
            nccl_comm.broadcast(xGPU, ndim, 0);
#endif         
         }
         gettimeofday(&t1_copy, NULL);
         time_copy = ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // axpy
            gettimeofday(&t0_inter, NULL);
            Hmmtask.inter(k, opaddr, alphas);
            gettimeofday(&t1_inter, NULL);
            // gemm on GPU
            gettimeofday(&t0_gemm, NULL);
            Hmmtask.kernel(k, ptrs);
            gettimeofday(&t1_gemm, NULL);
            // reduction
            gettimeofday(&t0_reduction, NULL);
            Hmmtask.reduction(k, xGPU, ptrs[6], yGPU, dev_red);
            gettimeofday(&t1_reduction, NULL);
            // timing
            time_inter += ((double)(t1_inter.tv_sec - t0_inter.tv_sec) 
                  + (double)(t1_inter.tv_usec - t0_inter.tv_usec)/1000000.0);
            time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                  + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
            time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                  + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
         } // k

         // copy yGPU to yCPU
         gettimeofday(&t0_copy, NULL);
         if(!ifnccl){
            GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#ifdef NCCL
         }else{
            nccl_comm.reduce(yGPU, ndim, 0);
            if(rank==0) GPUmem.to_cpu(yCPU, yGPU, ndim*sizeof(Tm));
#endif
         }
         gettimeofday(&t1_copy, NULL);
         time_copy += ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "preprocess_Hx_batchDirectGPUSingle: t[copy,inter,gemm,reduction]="
                      << time_copy << "," << time_inter << "," << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
            oper_timer.sigma.analysis();
         }
      }

} // ctns

#endif

#endif
