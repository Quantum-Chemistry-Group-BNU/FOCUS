#ifndef PREPROCESS_RENORM2_H
#define PREPROCESS_RENORM2_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"

namespace ctns{

   template <typename Tm> 
      void preprocess_renorm2(Tm* y,
            const Tm* x,
            const int& size,
            const int& rank,
            const size_t& ndim,
            const size_t& blksize,
            Rlist2<Tm>& Rlst2,
            Tm** opaddr){
         const bool debug = false;
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_renorm2"
               << " mpisize=" << size 
               << " maxthreads=" << maxthreads
               << std::endl;
         }

         // initialization
         memset(y, 0, ndim*sizeof(Tm));

#ifdef _OPENMP
#pragma omp parallel
         {
#endif

            Tm* work = new Tm[blksize*3];
            for(int i=0; i<Rlst2.size(); i++){
               memset(work, 0, blksize*sizeof(Tm));
#ifdef _OPENMP
#pragma omp for schedule(dynamic) nowait
#endif
               for(int j=0; j<Rlst2[i].size(); j++){
                  auto& Rblk = Rlst2[i][j];
                  Tm* wptr = &work[blksize];
                  Rblk.kernel(x, opaddr, wptr);
                  // save to local memory
                  linalg::xaxpy(Rblk.size, Rblk.coeff, wptr, work);
               } // j
               if(Rlst2[i].size()>0){
                  const auto& Rblk = Rlst2[i][0];
#ifdef _OPENMP
#pragma omp critical
#endif
                  linalg::xaxpy(Rblk.size, 1.0, work, y+Rblk.offrop);
               }
            } // i
            delete[] work;

#ifdef _OPENMP
         }
#endif

      }

} // ctns

#endif
