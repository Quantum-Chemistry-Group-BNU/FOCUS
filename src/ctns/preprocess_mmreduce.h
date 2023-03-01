#ifndef PREPROCESS_MMREDUCE_H
#define PREPROCESS_MMREDUCE_H

#ifdef GPU
#include "../gpu/gpu_blas.h"
#endif

namespace ctns{

   template <typename Tm>
      struct MMreduce{
         public:
            void reduction(Tm* workspace, Tm* y, const int iop) const;
         public:
            // size,ndim should be less than 2GB, otherwise calling xaxpy is problematic
            int size = 0, ndim = 0, offset = 0;
            size_t offout = 0; 
            std::vector<Tm> coeff;
      };

   // y[ndim] += \sum_i a_i*y_i[ndim]
   template <typename Tm>
      void MMreduce<Tm>::reduction(Tm* workspace, Tm* y, const int iop) const{
         const Tm alpha = 1.0, beta = 1.0; // accumulate
         const int INCX = 1, INCY = 1;
         Tm* yout = y + offout;
         if(iop == 0){
            /*
               for(int i=0; i<size; i++){
               Tm* yptr = workspace + i*offset;
               linalg::xaxpy(ndim, coeff[i], yptr, yout);
               }
               */
            linalg::xgemv("N", &ndim, &size, &alpha, workspace, &offset, 
                  coeff.data(), &INCX, &beta, yout, &INCY);
#ifdef GPU
         }else if(iop == 1){
            /*
               for(int i=0; i<size; i++){
               Tm* yptr = workspace + i*offset;
               linalg::xaxpy_magma(ndim, coeff[i], yptr, yout); 
               }
               */
            linalg::xgemv_magma("N", &ndim, &size, &alpha, workspace, &offset, 
                  coeff.data(), &INCX, &beta, yout, &INCY);
#endif
         }else{
            std::cout << "error: no such option in MMreduce<Tm>::reduction iop=" << iop << std::endl;
            exit(1);
         }
      }

} // ctns

#endif
