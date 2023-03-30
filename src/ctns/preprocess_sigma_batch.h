#ifndef PREPROCESS_SIGMA_BATCH_H
#define PREPROCESS_SIGMA_BATCH_H

#include "preprocess_hinter.h"
#include "preprocess_hmu.h"
#include "preprocess_hmmtask.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batch(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtasks<Tm>& Hmmtasks,
            Tm** opaddr,
            Tm* workspace,
            double& t_kernel_ibond,
            double& t_reduction_ibond){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batch"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

         // initialization
         memset(y, 0, ndim*sizeof(Tm));

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.sigma_start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // gemm
               gettimeofday(&t0_gemm, NULL);
               Hmmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               Hmmtask.reduction(k, ptrs[6], y);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
            } // k
         } // i
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batch: t[gemm,reduction]="
               << time_gemm << "," << time_reduction 
               << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
               << std::endl;
            oper_timer.sigma_analysis();
         }
         t_kernel_ibond = time_gemm;
         t_reduction_ibond = time_reduction;
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batch2(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtasks<Tm>& Hmmtasks,
            Tm** opaddr,
            Tm* workspace,
            Tm* alphas){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batch2"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_axpy, t1_axpy;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

         // initialization
         memset(y, 0, ndim*sizeof(Tm));

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.sigma_start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // axpy
               gettimeofday(&t0_axpy, NULL);
               Hmmtask.inter(k, opaddr, alphas);
               gettimeofday(&t1_axpy, NULL);
               // gemm
               gettimeofday(&t0_gemm, NULL);
               Hmmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               Hmmtask.reduction(k, ptrs[6], y);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_inter += ((double)(t1_axpy.tv_sec - t0_axpy.tv_sec) 
                     + (double)(t1_axpy.tv_usec - t0_axpy.tv_usec)/1000000.0);
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
            } // k
         } // i
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batch2: t[inter,gemm,reduction]="
               << time_inter << "," << time_gemm << "," << time_reduction 
               << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
               << std::endl;
            oper_timer.sigma_analysis();
         }
      }

} // ctns

#endif
