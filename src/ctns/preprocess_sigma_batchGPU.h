#ifndef PREPROCESS_SIGMA_BATCHGPU_H
#define PREPROCESS_SIGMA_BATCHGPU_H

/*
#include "preprocess_inter.h"
#include "preprocess_hmu.h"
#include "preprocess_batch.h"

namespace ctns{

// for Davidson diagonalization
template <typename Tm> 
void preprocess_Hx_batch_GPU(Tm* yCPU,
	                     const Tm* xCPU,
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
      std::cout << "ctns::preprocess_Hx_batchGPU"
	        << " mpisize=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }

   // initialization
   Tm* x = workspace[0];
   Tm* y = workspace[ndim];
   size_t offset = 2*ndim;

   // GPU: copy x vector (dimension=ndim)
   
   memset(y, 0, ndim*sizeof(Tm));

   Tm* ptrs[7];
   ptrs[0] = opaddr[0];
   ptrs[1] = opaddr[1];
   ptrs[2] = opaddr[2];
   ptrs[3] = opaddr[3];
   ptrs[4] = opaddr[4];
   ptrs[5] = x;
   ptrs[6] = &workspace[offset];

   // loop over nonzero blocks
   for(int i=0; i<mmtasks.size(); i++){
      auto& mmtask = mmtasks[i];
      for(int k=0; k<mmtask.nbatch; k++){
         // gemm on GPU
	 mmtask.kernel(k, ptrs);
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

   // GPU: copy y vector back to yCPU
   
   // add const term
   if(rank == 0) linalg::xaxpy(ndim, scale, xCPU, yCPU);
}


} // ctns
*/

#endif
