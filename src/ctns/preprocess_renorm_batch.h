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

         double time_gemm=0.0;
         double time_reduction=0.0;
         
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.renorm.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Rmmtasks.size(); i++){
            auto& Rmmtask = Rmmtasks[i];
            cost += Rmmtask.cost;
            for(int k=0; k<Rmmtask.nbatch; k++){
               // gemm
               auto t0gemm = tools::get_time();
               Rmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Rmmtask.reduction(k, x, ptrs[6], y);
               auto t1reduction = tools::get_time();
               // timing
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batch: t[gemm,reduction]="
               << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchDirect(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtasks<Tm>& Rmmtasks,
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
            std::cout << "ctns::preprocess_renorm_batchDirect"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;
         
         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.renorm.start();
         // loop over nonzero blocks
         double cost = 0.0;
         for(int i=0; i<Rmmtasks.size(); i++){
            auto& Rmmtask = Rmmtasks[i];
            cost += Rmmtask.cost;
            for(int k=0; k<Rmmtask.nbatch; k++){
               // inter 
               auto t0inter = tools::get_time();
               Rmmtask.inter(k, opaddr, alphas);
               auto t1inter = tools::get_time();
               // gemm
               auto t0gemm = tools::get_time();
               Rmmtask.kernel(k, ptrs);
               auto t1gemm = tools::get_time();
               // reduction
               auto t0reduction = tools::get_time();
               Rmmtask.reduction(k, x, ptrs[6], y);
               auto t1reduction = tools::get_time();
               // timing
               time_inter += tools::get_duration(t1inter-t0inter);
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batchDirect: t[inter,gemm,reduction]="
               << time_inter << "," << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchSingle(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtask<Tm>& Rmmtask,
            Tm** opaddr,
            Tm* workspace){
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         const bool debug = false;
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_renorm_batchSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.renorm.start();
         // loop over nonzero blocks
         double cost = Rmmtask.cost;
         for(int k=0; k<Rmmtask.nbatch; k++){
            // gemm
            auto t0gemm = tools::get_time();
            Rmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Rmmtask.reduction(k, x, ptrs[6], y);
            auto t1reduction = tools::get_time();
            // timing
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batchSingle: t[gemm,reduction]="
               << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchDirectSingle(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtask<Tm>& Rmmtask,
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
            std::cout << "ctns::preprocess_renorm_batchDirectSingle"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[7];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(x);
         ptrs[6] = workspace;

         oper_timer.renorm.start();
         // loop over nonzero blocks
         double cost = Rmmtask.cost;
         for(int k=0; k<Rmmtask.nbatch; k++){
            // inter 
            auto t0inter = tools::get_time();
            Rmmtask.inter(k, opaddr, alphas);
            auto t1inter = tools::get_time();
            // gemm
            auto t0gemm = tools::get_time();
            Rmmtask.kernel(k, ptrs);
            auto t1gemm = tools::get_time();
            // reduction
            auto t0reduction = tools::get_time();
            Rmmtask.reduction(k, x, ptrs[6], y);
            auto t1reduction = tools::get_time();
            // timing
            time_inter += tools::get_duration(t1inter-t0inter);
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         // timing
         if(rank == 0){
            std::cout << "preprocess_renorm_batchDirectSingle: t[inter,gemm,reduction]="
               << time_inter << "," << time_gemm << "," << time_reduction 
               << std::endl;
            oper_timer.renorm.analysis();
         }
      }

} // ctns

#endif
