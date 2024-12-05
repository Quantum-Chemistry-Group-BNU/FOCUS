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
            Tm* workspace){
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

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Hmmtasks.size(); i++){
            auto& Hmmtask = Hmmtasks[i];
            cost += Hmmtask.cost;
            for(int k=0; k<Hmmtask.nbatch; k++){
               // gemm
               auto t0gemm = tools::get_time();
               Hmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Hmmtask.reduction(k, x, ptrs[6], y);
               auto t1reduction = tools::get_time();
               // timing
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batch: T(gemm,reduction)="
               << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchDirect(Tm* y,
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
            std::cout << "ctns::preprocess_Hx_batchDirect"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

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
               // gemm
               auto t0gemm = tools::get_time();
               Hmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Hmmtask.reduction(k, x, ptrs[6], y);
               auto t1reduction = tools::get_time();
               // timing
               time_inter += tools::get_duration(t1inter-t0inter);
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batchDirect: T(inter,gemm,reduction)="
               << time_inter << "," << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchSingle(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtask<Tm>& Hmmtask,
            Tm** opaddr,
            Tm* workspace){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx_batchSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_gemm=0.0;
         double time_reduction=0.0;

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

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // gemm
            auto t0gemm = tools::get_time();
            Hmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Hmmtask.reduction(k, x, ptrs[6], y);
            auto t1reduction = tools::get_time();
            // timing
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batchSingle: T(gemm,reduction)="
               << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.sigma.analysis();
         }
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx_batchDirectSingle(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            HMMtask<Tm>& Hmmtask,
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
            std::cout << "ctns::preprocess_Hx_batchDirectSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

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

         oper_timer.sigma.start();
         // loop over nonzero blocks
         double cost = Hmmtask.cost;
         for(int k=0; k<Hmmtask.nbatch; k++){
            // inter
            auto t0inter = tools::get_time();
            Hmmtask.inter(k, opaddr, alphas);
            auto t1inter = tools::get_time();
            // gemm
            auto t0gemm = tools::get_time();
            Hmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Hmmtask.reduction(k, x, ptrs[6], y);
            auto t1reduction = tools::get_time();
            // timing
            time_inter += tools::get_duration(t1inter-t0inter);
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k
         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);

         // timing
         if(rank == 0){
            std::cout << "preprocess_Hx_batchDirectSingle: T(inter,gemm,reduction)="
               << time_inter << "," << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.sigma.analysis();
         }
      }

} // ctns

#endif
