#ifndef PREPROCESS_MMREDUCE_H
#define PREPROCESS_MMREDUCE_H

#ifdef GPU
#include "../gpu/gpu_blas.h"
#endif

namespace ctns{

   template <typename Tm>
      struct MMreduce{
         public:
            void reduction(Tm* workspace, Tm* y, const int iop);
         public:
            // size,ndim should be less than 2GB, otherwise calling xaxpy is problematic
            int size = 0, ndim = 0, offset = 0;
            size_t offout = 0; 
            std::vector<Tm> coeff;
      };

   // y[ndim] += \sum_i a_i*y_i[ndim]
   template <typename Tm>
      void MMreduce<Tm>::reduction(Tm* workspace, Tm* y, const int iop){
         const Tm alpha = 1.0, beta = 0.0;
         const int  INCX = 1, INCY = 1;
         Tm* yptr = y + offout;
         if(iop == 0){
            linalg::xgemv("N", &ndim, &size, &alpha, workspace, &offset, 
                  coeff.data(), &INCX, &beta, yptr, &INCY);
#ifdef GPU
         }else if(iop == 1){
            /*
            linalg::xgemv_magma("N", &ndim, &size, &alpha, workspace, &offset, 
                  coeff.data(), &INCX, &beta, yptr, &INCY);
               for(int i=0; i<size; i++){
               linalg::xaxpy_magma(ndim, alpha[i], yptr[i], y+offout); 
               }
            */
#endif
         }else{
            std::cout << "error: no such option in MMreduce<Tm>::reduction iop=" << iop << std::endl;
            exit(1);
         }
      }

} // ctns

#endif
