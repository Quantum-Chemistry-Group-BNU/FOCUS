#ifdef GPU

#ifndef PREPROCESS_SIGMA_BATCH2GPU_H
#define PREPROCESS_SIGMA_BATCH2GPU_H

#include "preprocess_hinter.h"
#include "preprocess_hmu.h"
#include "preprocess_mmtask.h"
#include "../gpu/gpu_env.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batch2GPU(Tm* yCPU,
            const Tm* xCPU,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            MMtasks<Tm>& mmtasks,
            Tm** opaddr,
            Tm* dev_workspace,
            Tm* alphas,
            double& t_kernel_ibond,
            double& t_reduction_ibond
            ){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batch2GPU"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         // initialization
         Tm* x = &dev_workspace[0];
         Tm* y = &dev_workspace[ndim];
         size_t offset = 2*ndim;

         // GPU: copy x vector (dimension=ndim)
         double time_copy=0.0;
         double time_axpy=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_copy, t1_copy;
         struct timeval t0_axpy, t1_axpy;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

         gettimeofday(&t0_copy, NULL);
#ifdef USE_HIP
         // from xCPU to x
         HIP_CHECK(hipMemcpy(x, xCPU,ndim*sizeof(Tm), hipMemcpyHostToDevice));
         // memset yGPU
         HIP_CHECK(hipMemset(y, 0, ndim*sizeof(Tm)));
#else
         CUDA_CHECK(cudaMemcpy(x, xCPU,ndim*sizeof(Tm), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemset(y, 0, ndim*sizeof(Tm)));
#endif //USE_HIP
         gettimeofday(&t1_copy, NULL);
         time_copy = ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = x;
         ptrs[6] = &dev_workspace[offset];

         oper_timer.sigma_start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<mmtasks.size(); i++){
            auto& mmtask = mmtasks[i];
            cost += mmtask.cost;
            for(int k=0; k<mmtask.nbatch; k++){
               // axpy
               gettimeofday(&t0_axpy, NULL);
               mmtask.inter(k, opaddr, alphas);
               gettimeofday(&t1_axpy, NULL);
               // gemm on GPU
               gettimeofday(&t0_gemm, NULL);
               mmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               mmtask.reduction(k, ptrs[6], y);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_axpy += ((double)(t1_axpy.tv_sec - t0_axpy.tv_sec) 
                     + (double)(t1_axpy.tv_usec - t0_axpy.tv_usec)/1000000.0);
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);

            } // k
         } // i

         // copy yGPU to yCPU
         gettimeofday(&t0_copy, NULL);
#ifdef USE_HIP
         HIP_CHECK(hipMemcpy(yCPU, y, ndim*sizeof(Tm), hipMemcpyDeviceToHost));
#else
         CUDA_CHECK(cudaMemcpy(yCPU, y, ndim*sizeof(Tm), cudaMemcpyDeviceToHost));
#endif
         gettimeofday(&t1_copy, NULL);
         time_copy += ((double)(t1_copy.tv_sec - t0_copy.tv_sec) 
               + (double)(t1_copy.tv_usec - t0_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "preprocess_Hx_batch2GPU: t[axpy,gemm,reduction]="
                      << time_axpy << "," << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
            oper_timer.sigma_analysis();
         }
         t_kernel_ibond += time_gemm;
         t_reduction_ibond += time_reduction;
      }

} // ctns

#endif

#endif
