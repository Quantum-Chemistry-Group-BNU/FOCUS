#ifndef PREPROCESS_SIGMA_H
#define PREPROCESS_SIGMA_H

#include "preprocess_inter.h"
#include "preprocess_hmu.h"

namespace ctns{

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            const size_t& blksize,
            Hxlist<Tm>& Hxlst,
            Tm** opaddr){
         const bool debug = false;
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         // initialization
         memset(y, 0, ndim*sizeof(Tm));

         // compute Y[I] = \sum_J H[I,J] X[J]
#ifdef _OPENMP

#pragma omp parallel
         {
            Tm* yi = new Tm[ndim];
            memset(yi, 0, ndim*sizeof(ndim));

            Tm* work = new Tm[blksize*2];
#pragma omp for schedule(dynamic) nowait
            for(int i=0; i<Hxlst.size(); i++){
               auto& Hxblk = Hxlst[i];
               Tm* wptr = work;
               Hxblk.kernel(x, opaddr, wptr);
               Tm* rptr = &work[Hxblk.offres];
               linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, yi+Hxblk.offout);
            } // i
            delete[] work;

#pragma omp critical
            linalg::xaxpy(ndim, 1.0, yi, y);

            delete[] yi;
         }

#else

         Tm* work = new Tm[blksize*2];
         for(int i=0; i<Hxlst.size(); i++){
            auto& Hxblk = Hxlst[i];
            Tm* wptr = work;
            Hxblk.kernel(x, opaddr, wptr);
            Tm* rptr = &work[Hxblk.offres];
            linalg::xaxpy(Hxblk.size, Hxblk.coeff, rptr, y+Hxblk.offout);
         } // i
         delete[] work;

#endif

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
      }

} // ctns

#endif
