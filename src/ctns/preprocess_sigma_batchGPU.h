#ifdef GPU

#ifndef PREPROCESS_SIGMA_BATCHGPU_H
#define PREPROCESS_SIGMA_BATCHGPU_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"
#include "preprocess_mmtask.h"
#include "../gpu/gpu_env.h"

#include "time.h"
#include "sys/time.h"

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
            const bool debug = false;
#ifdef _OPENMP
            int maxthreads = omp_get_max_threads();
#else
            int maxthreads = 1;
#endif
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
            double time_cost_gemm=0.0;
            double time_cost_gemm_kernel=0.0;
            double time_cost_gemm_copy=0.0;
            double time_cost_gemm_reduction=0.0;
            struct timeval t0_time_copy, t1_time_copy;
            struct timeval t0_time_gemm, t1_time_gemm;
            struct timeval t0_time_gemm_kernel, t1_time_gemm_kernel;
            struct timeval t0_time_gemm_copy, t1_time_gemm_copy;
            struct timeval t0_time_gemm_reduction, t1_time_gemm_reduction;

            // from xCPU to x
            gettimeofday(&t0_time_copy, NULL);
#if defined(USE_CUDA_OPERATION)
            CUDA_CHECK(cudaMemcpy(x, xCPU,ndim*sizeof(Tm), cudaMemcpyHostToDevice));
#else
                magma_dsetvector(ndim, (double*)xCPU, 1, (double*)x,  1,  magma_queue);
#endif
            gettimeofday(&t1_time_copy, NULL);
            
            time_cost_copy = ((double)(t1_time_copy.tv_sec - t0_time_copy.tv_sec) + (double)(t1_time_copy.tv_usec - t0_time_copy.tv_usec)/1000000.0);

            // TODOs: memset yGPU
            cudaMemset(y, 0, ndim*sizeof(Tm));

            Tm* ptrs[7];
            ptrs[0] = opaddr[0];
            ptrs[1] = opaddr[1];
            ptrs[2] = opaddr[2];
            ptrs[3] = opaddr[3];
            ptrs[4] = opaddr[4];
            ptrs[5] = x;
            ptrs[6] = &dev_workspace[offset];

            double flops_G=0.0;

            gettimeofday(&t0_time_gemm, NULL);
            // loop over nonzero blocks
            for(int i=0; i<mmtasks.size(); i++){
                auto& mmtask = mmtasks[i];
                for(int k=0; k<mmtask.nbatch; k++){
                    
                    
                    double flops_tt=0.0;
                    // gemm on GPU
                    gettimeofday(&t0_time_gemm_kernel, NULL);
                    mmtask.kernel(k, ptrs, flops_tt);
                    gettimeofday(&t1_time_gemm_kernel, NULL);
                    flops_G += flops_tt;

                    // reduction
                    gettimeofday(&t0_time_gemm_reduction, NULL);
                    mmtask.reduction(k, ptrs[6], y, 1);
                    gettimeofday(&t1_time_gemm_reduction, NULL);

                    time_cost_gemm_kernel += ((double)(t1_time_gemm_kernel.tv_sec - t0_time_gemm_kernel.tv_sec) + (double)(t1_time_gemm_kernel.tv_usec - t0_time_gemm_kernel.tv_usec)/1000000.0);
                    time_cost_gemm_reduction += ((double)(t1_time_gemm_reduction.tv_sec - t0_time_gemm_reduction.tv_sec) + (double)(t1_time_gemm_reduction.tv_usec - t0_time_gemm_reduction.tv_usec)/1000000.0);

                } // k
            } // i

            gettimeofday(&t1_time_gemm, NULL);
            time_cost_gemm = ((double)(t1_time_gemm.tv_sec - t0_time_gemm.tv_sec) + (double)(t1_time_gemm.tv_usec - t0_time_gemm.tv_usec)/1000000.0);

            //std::cout<<"time_cost_copy xcpu hosttodevice size ndim ="<<time_cost_copy<<"; time_cost_gemm  kernel+reduction total="<<time_cost_gemm<<std::endl;
            std::cout<<"time_cost_gemm_kernel="<<time_cost_gemm_kernel<<std::endl;
            std::cout<<"time_cost_gemm_reduction="<<time_cost_gemm_reduction<<std::endl;
            //std::cout<<"time_sum kernel+reduction="<<time_cost_gemm_kernel+time_cost_gemm_copy+time_cost_gemm_reduction<<std::endl;
            std::cout<<"gflops=2*m*n*k/time = kernel/time="<<flops_G/time_cost_gemm_kernel<<" flops_G:"<<flops_G<<std::endl;

            t_kernel_ibond = time_cost_gemm_kernel;
            t_reduction_ibond = time_cost_gemm_reduction;

            // TODOs: copy yGPU to yCPU
            gettimeofday(&t0_time_gemm_copy, NULL);
#if defined(USE_CUDA_OPERATION)
            CUDA_CHECK(cudaMemcpy(yCPU,y, ndim*sizeof(Tm), cudaMemcpyDeviceToHost));
#else
            magma_dgetvector(ndim, (double*)y, 1, (double*)yCPU,  1,  magma_queue);
#endif
            gettimeofday(&t1_time_gemm_copy, NULL);
            time_cost_gemm_copy += ((double)(t1_time_gemm_copy.tv_sec - t0_time_gemm_copy.tv_sec) + (double)(t1_time_gemm_copy.tv_usec - t0_time_gemm_copy.tv_usec)/1000000.0);
            //std::cout<<"time_cost_gemm_copy yGPU devicetohost size ndim ="<<time_cost_gemm_copy<<std::endl;

            // add const term
            if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);
        }


} // ctns

#endif

#endif
