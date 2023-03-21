#ifndef PREPROCESS_RENORM_BATCH_H
#define PREPROCESS_RENORM_BATCH_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"
#include "preprocess_rmmtask.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   template <typename Tm> 
      void preprocess_renorm_batch(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            const size_t& blksize,
            Rlist2<Tm>& Rlst2,
            RMMtasks<Tm>& Rmmtasks,
            Tm** opaddr,
            Tm* workspace){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_renorm_batch"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

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

         double time_cost_gemm_kernel=0.0;
         double time_cost_gemm_reduction=0.0;
         struct timeval t0_time_gemm_kernel, t1_time_gemm_kernel;
         struct timeval t0_time_gemm_reduction, t1_time_gemm_reduction;

         oper_timer.renorm_start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Rmmtasks.size(); i++){
            auto& Rmmtask = Rmmtasks[i];
            cost += Rmmtask.cost;
            for(int k=0; k<Rmmtask.nbatch; k++){
               // gemm
               gettimeofday(&t0_time_gemm_kernel, NULL);
               Rmmtask.kernel(k, ptrs);
               gettimeofday(&t1_time_gemm_kernel, NULL);
               // reduction
               gettimeofday(&t0_time_gemm_reduction, NULL);
               Rmmtask.reduction(k, ptrs[6], y, 0);
               gettimeofday(&t1_time_gemm_reduction, NULL);
               // timing
               time_cost_gemm_kernel += ((double)(t1_time_gemm_kernel.tv_sec - t0_time_gemm_kernel.tv_sec) 
                     + (double)(t1_time_gemm_kernel.tv_usec - t0_time_gemm_kernel.tv_usec)/1000000.0);
               time_cost_gemm_reduction += ((double)(t1_time_gemm_reduction.tv_sec - t0_time_gemm_reduction.tv_sec) 
                     + (double)(t1_time_gemm_reduction.tv_usec - t0_time_gemm_reduction.tv_usec)/1000000.0);
            } // k
         } // i

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batch: t[gemm,reduction]="
                      << time_cost_gemm_kernel << ","
                      << time_cost_gemm_reduction << " cost="
                      << cost << " flops[gemm]=" << cost/time_cost_gemm_kernel
                      << std::endl;
            oper_timer.renorm_analysis();
         }
      }

} // ctns

#endif
