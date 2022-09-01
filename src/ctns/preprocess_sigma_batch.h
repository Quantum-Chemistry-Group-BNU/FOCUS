#ifndef PREPROCESS_SIGMA_BATCH_H
#define PREPROCESS_SIGMA_BATCH_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"
#include "preprocess_mmtask.h"

#include "time.h"
#include "sys/time.h"

namespace ctns{

template <typename Tm, typename QTm>
void preprocess_formulae_sigma_batch(const oper_dictmap<Tm>& qops_dict,
				       const std::map<std::string,int>& oploc,
		 	               const symbolic_task<Tm>& H_formulae,
			               const QTm& wf,
				       intermediates<Tm>& inter,
   			               Hxlist2<Tm>& Hxlst2,
				       size_t& blksize,
				       double& cost,
				       const bool debug){
   auto t0 = tools::get_time();

   // 1. form intermediate operators 
   inter.init(qops_dict,H_formulae,debug);
   auto ta = tools::get_time();

   // 2. preprocess formulae to Hmu
   int hsize = H_formulae.size();
   std::vector<Hmu_ptr<Tm>> Hmu_vec(hsize);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].init(it, H_formulae, qops_dict, inter, oploc); 
   } // it
   auto tb = tools::get_time();

   // 3. from Hmu to expanded block forms
   blksize = 0;
   cost = 0.0;
   int nnzblk = wf.info._nnzaddr.size();
   Hxlst2.resize(nnzblk);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, true);
   }
   auto tc = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      std::cout << "T(inter/Hmu/Hxlist/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_sigma_batch", t0, t1);
   }
}

// for Davidson diagonalization
template <typename Tm> 
void preprocess_Hx_batch(Tm* y,
	                 const Tm* x,
		         const Tm& scale,
		         const int& size,
	                 const int& rank,
		         const size_t& ndim,
	                 const size_t& blksize,
			 Hxlist2<Tm>& Hxlst2,
			 MMtasks<Tm>& mmtasks,
		         Tm** opaddr,
		         Tm* workspace,
                	 double& t_kernel_ibond,
                	 double& t_reduction_ibond){
   const bool debug = false;
#ifdef _OPENMP
            int maxthreads = omp_get_max_threads();
#else
            int maxthreads = 1;
#endif
            if(rank == 0 && debug){
                std::cout << "ctns::preprocess_Hx_batch"
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

            double flops_G=0.0;

            // loop over nonzero blocks
            for(int i=0; i<mmtasks.size(); i++){
                auto& mmtask = mmtasks[i];
                for(int k=0; k<mmtask.nbatch; k++){
                    double flops_tt=0.0;
                    // gemm
                    gettimeofday(&t0_time_gemm_kernel, NULL);
                    mmtask.kernel(k, ptrs, flops_tt);
                    gettimeofday(&t1_time_gemm_kernel, NULL);
                    flops_G += flops_tt;
                    // reduction
                    gettimeofday(&t0_time_gemm_reduction, NULL);
                    mmtask.reduction(k, ptrs[6], y, 0);
                    gettimeofday(&t1_time_gemm_reduction, NULL);
                    time_cost_gemm_kernel += ((double)(t1_time_gemm_kernel.tv_sec - t0_time_gemm_kernel.tv_sec) + (double)(t1_time_gemm_kernel.tv_usec - t0_time_gemm_kernel.tv_usec)/1000000.0);
                    time_cost_gemm_reduction += ((double)(t1_time_gemm_reduction.tv_sec - t0_time_gemm_reduction.tv_sec) + (double)(t1_time_gemm_reduction.tv_usec - t0_time_gemm_reduction.tv_usec)/1000000.0);
                } // k
            } // i
            std::cout<<"time_cost_gemm_kernel="<<time_cost_gemm_kernel<<std::endl;
            std::cout<<"time_cost_gemm_reduction="<<time_cost_gemm_reduction<<std::endl;
            //std::cout<<"time_sum above="<<time_cost_gemm_kernel+time_cost_gemm_reduction<<std::endl;
            std::cout<<"gflops=2*m*n*k/time = kernel/time="<<flops_G/time_cost_gemm_kernel<<" flops_G:"<<flops_G<<std::endl;
            t_kernel_ibond = time_cost_gemm_kernel;
            t_reduction_ibond = time_cost_gemm_reduction;

            // add const term
            if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
        }


} // ctns

#endif
