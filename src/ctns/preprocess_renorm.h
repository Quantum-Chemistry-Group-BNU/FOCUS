#ifndef PREPROCESS_RENORM_H
#define PREPROCESS_RENORM_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"

namespace ctns{

   template <typename Tm> 
      void preprocess_renorm(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            const size_t& blksize,
            Rlist<Tm>& Rlst,
            Tm** opaddr){
         const bool debug = false;
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_renorm"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

#ifndef _OPENMP

         Tm* work = new Tm[blksize*2];
         for(int i=0; i<Rlst.size(); i++){
            auto& Rblk = Rlst[i];
            Rblk.kernel(x, opaddr, work);
            linalg::xaxpy(Rblk.size, Rblk.coeff, work, y+Rblk.offrop);
         } // i
         delete[] work;

#else

         #pragma omp parallel
         {
            Tm* yi = new Tm[ndim];
            memset(yi, 0, ndim*sizeof(ndim));

            Tm* work = new Tm[blksize*2];
            #pragma omp for schedule(dynamic) nowait
            for(int i=0; i<Rlst.size(); i++){
               auto& Rblk = Rlst[i];
               Rblk.kernel(x, opaddr, work);
               linalg::xaxpy(Rblk.size, Rblk.coeff, work, yi+Rblk.offrop);
            } // i
            delete[] work;

            #pragma omp critical
            linalg::xaxpy(ndim, 1.0, yi, y);

            delete[] yi;
         }

#endif
      }

} // ctns

#endif
