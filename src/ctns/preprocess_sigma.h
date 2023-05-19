#ifndef PREPROCESS_SIGMA_H
#define PREPROCESS_SIGMA_H

#include "preprocess_hinter.h"
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
#ifndef _OPENMP

         Tm* work = new Tm[blksize*2];
         for(int i=0; i<Hxlst.size(); i++){
            auto& Hxblk = Hxlst[i];
            bool ifcal = Hxblk.kernel(x, opaddr, work);
            if(ifcal) linalg::xaxpy(Hxblk.size, Hxblk.coeff, work, y+Hxblk.offout);
         } // i
         delete[] work;

#else

#pragma omp parallel
         {
            Tm* yi = new Tm[ndim];
            memset(yi, 0, ndim*sizeof(ndim));

            Tm* work = new Tm[blksize*2];
#pragma omp for schedule(dynamic) nowait
            for(int i=0; i<Hxlst.size(); i++){
               auto& Hxblk = Hxlst[i];
               bool ifcal = Hxblk.kernel(x, opaddr, work);
               if(ifcal) linalg::xaxpy(Hxblk.size, Hxblk.coeff, work, yi+Hxblk.offout);
            } // i
            delete[] work;

#pragma omp critical
            linalg::xaxpy(ndim, 1.0, yi, y);

            delete[] yi;
         }

#endif

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
      }

   // for Davidson diagonalization
   template <typename Tm> 
      void preprocess_Hx2(Tm* y,
            const Tm* x,
            const Tm& scale,
            const int& size,
            const int& rank,
            const size_t& ndim,
            const size_t& blksize,
            Hxlist2<Tm>& Hxlst2,
            Tm** opaddr){
         const bool debug = false;
#ifdef _OPENMP
         int maxthreads = omp_get_max_threads();
#else
         int maxthreads = 1;
#endif
         if(rank == 0 && debug){
            std::cout << "ctns::preprocess_Hx2"
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
#endif

            Tm* work = new Tm[blksize*3];
            for(int i=0; i<Hxlst2.size(); i++){
               memset(work, 0, blksize*sizeof(Tm));
#ifdef _OPENMP
#pragma omp for schedule(dynamic) nowait
#endif
               for(int j=0; j<Hxlst2[i].size(); j++){
                  auto& Hxblk = Hxlst2[i][j];
                  Tm* wptr = &work[blksize];
                  bool ifcal = Hxblk.kernel(x, opaddr, wptr);
                  if(ifcal) linalg::xaxpy(Hxblk.size, Hxblk.coeff, wptr, work); // save to local memory
               } // j
               if(Hxlst2[i].size()>0){
                  const auto& Hxblk = Hxlst2[i][0];
#ifdef _OPENMP
#pragma omp critical
#endif
                  linalg::xaxpy(Hxblk.size, 1.0, work, y+Hxblk.offout);
               }
            } // i
            delete[] work;

#ifdef _OPENMP
         }
#endif

         // add const term
         if(rank == 0) linalg::xaxpy(ndim, scale, x, y);
      }


} // ctns

#endif
