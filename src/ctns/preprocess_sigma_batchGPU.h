#ifdef GPU

#ifndef PREPROCESS_SIGMA_BATCHGPU_H
#define PREPROCESS_SIGMA_BATCHGPU_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"
#include "preprocess_mmtask.h"
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
            const size_t& blksize,
            Hxlist2<Tm>& Hxlst2,
            MMtasks<Tm>& mmtasks,
            Tm** opaddr,
            Tm* dev_workspace,
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
            std::cout << "ctns::preprocess_Hx_batchGPU"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         // initialization
         Tm* x = &dev_workspace[0];
         Tm* y = &dev_workspace[ndim];
         size_t offset = 2*ndim;

         // GPU: copy x vector (dimension=ndim)
         double time_cost_copy=0.0;
         double time_cost_gemm_kernel=0.0;
         double time_cost_gemm_reduction=0.0;
         struct timeval t0_time_copy, t1_time_copy;
         struct timeval t0_time_gemm_kernel, t1_time_gemm_kernel;
         struct timeval t0_time_gemm_reduction, t1_time_gemm_reduction;

         gettimeofday(&t0_time_copy, NULL);
#ifdef USE_HIP
         // from xCPU to x
         HIP_CHECK(hipMemcpy(x, xCPU,ndim*sizeof(Tm), hipMemcpyHostToDevice));
         // memset yGPU
         HIP_CHECK(hipMemset(y, 0, ndim*sizeof(Tm)));
#else
         CUDA_CHECK(cudaMemcpy(x, xCPU,ndim*sizeof(Tm), cudaMemcpyHostToDevice));
         CUDA_CHECK(cudaMemset(y, 0, ndim*sizeof(Tm)));
#endif //USE_HIP
         gettimeofday(&t1_time_copy, NULL);
         time_cost_copy = ((double)(t1_time_copy.tv_sec - t0_time_copy.tv_sec) 
               + (double)(t1_time_copy.tv_usec - t0_time_copy.tv_usec)/1000000.0);

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = x;
         ptrs[6] = &dev_workspace[offset];

         oper_timer.start_Hx();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<mmtasks.size(); i++){
            auto& mmtask = mmtasks[i];
            cost += mmtask.cost;
            for(int k=0; k<mmtask.nbatch; k++){
               // gemm on GPU
               gettimeofday(&t0_time_gemm_kernel, NULL);
               mmtask.kernel(k, ptrs);
               gettimeofday(&t1_time_gemm_kernel, NULL);
               // reduction
               gettimeofday(&t0_time_gemm_reduction, NULL);
               mmtask.reduction(k, ptrs[6], y, 1);
               gettimeofday(&t1_time_gemm_reduction, NULL);
               // timing
               time_cost_gemm_kernel += ((double)(t1_time_gemm_kernel.tv_sec - t0_time_gemm_kernel.tv_sec) 
                     + (double)(t1_time_gemm_kernel.tv_usec - t0_time_gemm_kernel.tv_usec)/1000000.0);
               time_cost_gemm_reduction += ((double)(t1_time_gemm_reduction.tv_sec - t0_time_gemm_reduction.tv_sec) 
                     + (double)(t1_time_gemm_reduction.tv_usec - t0_time_gemm_reduction.tv_usec)/1000000.0);

            } // k
         } // i

         // copy yGPU to yCPU
         gettimeofday(&t0_time_copy, NULL);
#ifdef USE_HIP
         HIP_CHECK(hipMemcpy(yCPU, y, ndim*sizeof(Tm), hipMemcpyDeviceToHost));
#else
         CUDA_CHECK(cudaMemcpy(yCPU, y, ndim*sizeof(Tm), cudaMemcpyDeviceToHost));
#endif
         gettimeofday(&t1_time_copy, NULL);
         time_cost_copy += ((double)(t1_time_copy.tv_sec - t0_time_copy.tv_sec) 
               + (double)(t1_time_copy.tv_usec - t0_time_copy.tv_usec)/1000000.0);

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);

         // timing
         if(rank==0){
            std::cout << "--- preprocess_Hx_batchGPU ---" << std::endl;
            std::cout << "--- time_copy=" << time_cost_copy << std::endl;
            std::cout << "--- time_gemm_kernel=" << time_cost_gemm_kernel << std::endl;
            std::cout << "--- time_gemm_reduction=" << time_cost_gemm_reduction << std::endl;
            std::cout << "--- cost_gemm_kernel=" << cost
               << " flops=kernel/time=" << cost/time_cost_gemm_kernel
               << std::endl;
            oper_timer.analysis_Hx();
         }
         t_kernel_ibond += time_cost_gemm_kernel;
         t_reduction_ibond += time_cost_gemm_reduction;
      }

} // ctns

#endif

#endif
