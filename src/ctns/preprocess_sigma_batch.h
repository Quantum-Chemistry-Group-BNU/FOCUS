#ifndef PREPROCESS_SIGMA_BATCH_H
#define PREPROCESS_SIGMA_BATCH_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"
#include "preprocess_batch.h"

namespace ctns{

template <typename Tm, typename QTm>
size_t preprocess_formulae_sigma_batch(const oper_dictmap<Tm>& qops_dict,
				       const std::map<std::string,int>& oploc,
		 	               const symbolic_task<Tm>& H_formulae,
			               const QTm& wf,
				       intermediates<Tm>& inter,
   			               Hxlist2<Tm>& Hxlst2,
				       MMtasks<Tm>& mmtasks,
				       const int hxorder,
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
   size_t blksize = 0;
   double cost = 0.0;
   int nnzblk = wf.info._nnzaddr.size();
   Hxlst2.resize(nnzblk);
   for(int it=0; it<hsize; it++){
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, false);
      Hmu_vec[it].gen_Hxlist2(wf.info, Hxlst2, blksize, cost, true);
   }
   auto tc = tools::get_time();

   // 4. gen MMlst & reorder
   const size_t batchsize = 1000;
   mmtasks.resize(nnzblk);
   for(int i=0; i<nnzblk; i++){
      mmtasks[i].init(Hxlst2[i], batchsize, hxorder);
   } // i
   auto td = tools::get_time();

   if(debug){
      auto t1 = tools::get_time();
      size_t hxsize = 0;
      for(int i=0; i<nnzblk; i++){
	 hxsize += Hxlst2[i].size();
      }
      std::cout << "size(Hxlst2)=" << nnzblk
	        << " size(Hxlst)=" << hxsize
                << " size(formulae)=" << hsize
	        << " hxsize/nnzblk=" << hxsize/double(nnzblk)
		<< " cost=" << cost 
	        << std::endl;
      std::cout << "T(inter/Hmu/Hxlist/sort/tot)="
	        << tools::get_duration(ta-t0) << ","
	        << tools::get_duration(tb-ta) << ","
	        << tools::get_duration(tc-tb) << ","
	        << tools::get_duration(td-tc) << ","
		<< tools::get_duration(t1-t0) 
		<< std::endl;
      tools::timing("preprocess_formulae_sigma_batch", t0, t1);
   }
   return blksize;
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
		         Tm* workspace){
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

   // loop over nonzero blocks
   for(int i=0; i<mmtasks.size(); i++){
      auto& mmtask = mmtasks[i];
      for(int k=0; k<mmtask.nbatch; k++){
/*
         // gemm
         mmtask.mmbatch2[k][0].kernel(ptrs); // c2 
         mmtask.mmbatch2[k][1].kernel(ptrs); // c1
         mmtask.mmbatch2[k][2].kernel(ptrs); // r
         mmtask.mmbatch2[k][3].kernel(ptrs); // l
*/
         // reduction
	 size_t off = k*mmtask.batchsize;
	 size_t jlen = std::min(mmtask.totsize-off,mmtask.batchsize);
         for(int j=0; j<jlen; j++){
            int jdx = k*mmtask.batchsize+j;
            auto& Hxblk = Hxlst2[i][jdx];
	    Tm* rptr = &workspace[j*blksize*2+Hxblk.offres];
            linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, y+Hxblk.offout);
         }
      } // k
   } // i

   // add const term
   if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
}


} // ctns

#endif
