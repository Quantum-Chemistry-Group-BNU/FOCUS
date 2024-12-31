#ifdef GPU

#ifndef PREPROCESS_RENORM_BATCHGPU_H
#define PREPROCESS_RENORM_BATCHGPU_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"
#include "preprocess_rmmtask.h"
#include "../gpu/gpu_env.h"

#include "time.h"
#include "sys/time.h"
#include "oper_timer.h"

namespace ctns{

   template <typename Tm> 
      void preprocess_renorm_batchGPU(Tm* y,
            const Tm* xbra,
            const Tm* xket,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtasks<Tm>& Rmmtasks,
            Tm** opaddr,
            Tm* workspace,
            Tm* dev_red,
            const bool debug){
         
         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[8];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(xket);
         ptrs[6] = workspace;
         ptrs[7] = const_cast<Tm*>(xbra);

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
               Rmmtask.reduction(k, ptrs[6], y, dev_red);
               auto t1reduction = tools::get_time();
               // timing
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         if(debug){
            std::cout << "preprocess_renorm_batchGPU: T(gemm,reduction)="
                      << time_gemm << "," << time_reduction 
                      << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchDirectGPU(Tm* y,
            const Tm* xbra,
            const Tm* xket,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtasks<Tm>& Rmmtasks,
            Tm** opaddr,
            Tm* workspace,
            Tm* alphas,
            Tm* dev_red,
            const bool debug){

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[8];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(xket);
         ptrs[6] = workspace;
         ptrs[7] = const_cast<Tm*>(xbra);

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
               Rmmtask.reduction(k, ptrs[6], y, dev_red);
               auto t1reduction = tools::get_time();
               // timing
               time_inter += tools::get_duration(t1inter-t0inter);
               time_gemm += tools::get_duration(t1gemm-t0gemm);
               time_reduction += tools::get_duration(t1reduction-t0reduction);
            } // k
         } // i

         if(debug){
            std::cout << "preprocess_renorm_batchDirectGPU: T(inter,gemm,reduction)="
                      << time_inter << "," << time_gemm << "," << time_reduction 
                      << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchGPUSingle(Tm* y,
            const Tm* xbra,
            const Tm* xket,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtask<Tm>& Rmmtask,
            Tm** opaddr,
            Tm* workspace,
            Tm* dev_red,
            const bool debug){

         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[8];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(xket);
         ptrs[6] = workspace;
         ptrs[7] = const_cast<Tm*>(xbra);

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
            Rmmtask.reduction(k, ptrs[6], y, dev_red);
            auto t1reduction = tools::get_time();
            // timing
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         if(debug){
            std::cout << "preprocess_renorm_batchGPUSingle: T(gemm,reduction)="
                      << time_gemm << "," << time_reduction 
                      << std::endl;
            oper_timer.renorm.analysis();
         }
      }

   template <typename Tm> 
      void preprocess_renorm_batchDirectGPUSingle(Tm* y,
            const Tm* xbra,
            const Tm* xket,
            const int& size,
            const int& rank,
            const size_t& ndim,
            RMMtask<Tm>& Rmmtask,
            Tm** opaddr,
            Tm* workspace,
            Tm* alphas,
            Tm* dev_red,
            const bool debug){

         double time_inter=0.0;
         double time_gemm=0.0;
         double time_reduction=0.0;

         Tm* ptrs[8];
         ptrs[0] = opaddr[0];
         ptrs[1] = opaddr[1];
         ptrs[2] = opaddr[2];
         ptrs[3] = opaddr[3];
         ptrs[4] = opaddr[4];
         ptrs[5] = const_cast<Tm*>(xket);
         ptrs[6] = workspace;
         ptrs[7] = const_cast<Tm*>(xbra);

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
            Rmmtask.reduction(k, ptrs[6], y, dev_red);
            auto t1reduction = tools::get_time();
            // timing
            time_inter += tools::get_duration(t1inter-t0inter);
            time_gemm += tools::get_duration(t1gemm-t0gemm);
            time_reduction += tools::get_duration(t1reduction-t0reduction);
         } // k

         if(debug){
            std::cout << "preprocess_renorm_batchDirectGPUDirect: T(inter,gemm,reduction)="
                      << time_inter << "," << time_gemm << "," << time_reduction 
                      << std::endl;
            oper_timer.renorm.analysis();
         }
      }

} // ctns

#endif

#endif
