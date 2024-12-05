#ifndef PREPROCESS_RENORM_H
#define PREPROCESS_RENORM_H

#include "preprocess_rinter.h"
#include "preprocess_rmu.h"

namespace ctns{

   template <typename Tm> 
      void preprocess_renorm(Tm* y,
            const Tm* xbra,
            const Tm* xket,
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

         // init
         memset(y, 0, ndim*sizeof(Tm));

#ifdef _OPENMP
#pragma omp parallel
         {
#endif

            Tm* work = new Tm[blksize*2];
#ifdef _OPENMP
#pragma omp for schedule(dynamic) nowait
#endif
            for(int i=0; i<Rlst.size(); i++){
               auto& Rblk = Rlst[i];
               bool ifcal = Rblk.kernel(xbra, xket, opaddr, work);
#ifdef _OPENMP
#pragma omp critical
#endif
               if(ifcal) linalg::xaxpy(Rblk.size, Rblk.coeff, work, y+Rblk.offrop);
            } // i
            delete[] work;

#ifdef _OPENMP
         }
#endif
      }

} // ctns

#endif
