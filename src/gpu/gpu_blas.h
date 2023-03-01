#ifdef GPU

#ifndef GPU_BLAS_H
#define GPU_BLAS_H

#include <vector>
#include <complex>

#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

   // axpy
   inline void xaxpy_magma(const int N, const double alpha, 
         const double* X, double* Y){
      const int INCX = 1, INCY = 1;
      magma_daxpy(N, alpha, X, INCX, Y, INCY, magma_queue );
   }
   inline void xaxpy_magma(const int N, const std::complex<double> alpha, 
         const std::complex<double>* X, std::complex<double>* Y){
      const int INCX = 1, INCY = 1;
      magmaDoubleComplex alpha1{alpha.real(),alpha.imag()};
      magma_zaxpy(N, alpha1, (magmaDoubleComplex *)X, INCX, (magmaDoubleComplex *)Y, INCY, magma_queue);
   }

   // y = alpha*A*x + beta*y
   inline void xgemv_magma(const char* TRANSA, const int* M, const int* N,
         const double* alpha, const double* A, const int* LDA,
         const double* X, const int* INCX,
         const double* beta, double* Y, const int* INCY ){
      magma_trans_t transA =  MagmaNoTrans ;
      if(*TRANSA=='T'){
         transA = MagmaTrans;
      }else if (*TRANSA == 'C'){
         transA = MagmaConjTrans;
      }
      magma_dgemv(transA, *M, *N, *alpha, A, *LDA,
            X, *INCX, *beta, Y, *INCY, magma_queue);
   }
   inline void xgemv_magma(const char* TRANSA, const int* M, const int* N,
         const std::complex<double>* alpha, const std::complex<double>* A, const int* LDA,
         const std::complex<double>* X, const int* INCX,
         const std::complex<double>* beta, std::complex<double>* Y, const int* INCY){
      magma_trans_t transA =  MagmaNoTrans ;
      if(*TRANSA == 'T'){
         transA = MagmaTrans;
      }else if (*TRANSA == 'C'){
         transA = MagmaConjTrans;
      }
      magmaDoubleComplex alpha1{alpha->real(),alpha->imag()};
      magmaDoubleComplex beta1{beta->real(),beta->imag()};
      magma_zgemv(transA, *M, *N, alpha1, (magmaDoubleComplex *)A, *LDA,
            (magmaDoubleComplex *)X, *INCX, beta1, (magmaDoubleComplex *)Y, *INCY, magma_queue);
   }
} // linalg

#endif

#endif
