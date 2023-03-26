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

         double time_gemm=0.0;
         double time_reduction=0.0;
         struct timeval t0_gemm, t1_gemm;
         struct timeval t0_reduction, t1_reduction;

         oper_timer.renorm_start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Rmmtasks.size(); i++){
            auto& Rmmtask = Rmmtasks[i];
            cost += Rmmtask.cost;
            for(int k=0; k<Rmmtask.nbatch; k++){
               // gemm
               gettimeofday(&t0_gemm, NULL);
               Rmmtask.kernel(k, ptrs);
               gettimeofday(&t1_gemm, NULL);
               // reduction
               gettimeofday(&t0_reduction, NULL);
               Rmmtask.reduction(k, ptrs[6], y);
               gettimeofday(&t1_reduction, NULL);
               // timing
               time_gemm += ((double)(t1_gemm.tv_sec - t0_gemm.tv_sec) 
                     + (double)(t1_gemm.tv_usec - t0_gemm.tv_usec)/1000000.0);
               time_reduction += ((double)(t1_reduction.tv_sec - t0_reduction.tv_sec) 
                     + (double)(t1_reduction.tv_usec - t0_reduction.tv_usec)/1000000.0);
            } // k
         } // i

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batch: t[gemm,reduction]="
                      << time_gemm << "," << time_reduction 
                      << " cost=" << cost << " flops[gemm]=" << cost/time_gemm
                      << std::endl;
            oper_timer.renorm_analysis();
         }
      }

} // ctns

#endif
