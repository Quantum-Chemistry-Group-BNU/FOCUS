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
      if(TRANSA[0]=='T'){
         transA = MagmaTrans;
      }else if (TRANSA[0] == 'C'){
         transA = MagmaConjTrans;
      }
      double* dev_X = nullptr;
      dev_X = (double*)gpumem.allocate((*N)*sizeof(double));
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_X, X, (*N)*sizeof(double), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_X, X, (*N)*sizeof(double), cudaMemcpyHostToDevice));
#endif// USE_HIP
      magma_dgemv(transA, *M, *N, *alpha, A, *LDA,
            dev_X, *INCX, *beta, Y, *INCY, magma_queue);

      gpumem.deallocate(dev_X, (*N)*sizeof(double));
   }
   inline void xgemv_magma(const char* TRANSA, const int* M, const int* N,
         const std::complex<double>* alpha, const std::complex<double>* A, const int* LDA,
         const std::complex<double>* X, const int* INCX,
         const std::complex<double>* beta, std::complex<double>* Y, const int* INCY){
      magma_trans_t transA =  MagmaNoTrans ;
      if(TRANSA[0] == 'T'){
         transA = MagmaTrans;
      }else if (TRANSA[0] == 'C'){
         transA = MagmaConjTrans;
      }
      magmaDoubleComplex* dev_X = nullptr;
      dev_X = (magmaDoubleComplex*)gpumem.allocate((*N)*sizeof(magmaDoubleComplex));
#ifdef USE_HIP
            HIP_CHECK(hipMemcpy(dev_X, X, (*N)*sizeof(magmaDoubleComplex), hipMemcpyHostToDevice));
#else
            CUDA_CHECK(cudaMemcpy(dev_X, X, (*N)*sizeof(magmaDoubleComplex), cudaMemcpyHostToDevice));
#endif// USE_HIP
      magmaDoubleComplex alpha1{alpha->real(),alpha->imag()};
      magmaDoubleComplex beta1{beta->real(),beta->imag()};
      magma_zgemv(transA, *M, *N, alpha1, (magmaDoubleComplex *)A, *LDA,
            (magmaDoubleComplex *)dev_X, *INCX, beta1, (magmaDoubleComplex *)Y, *INCY, magma_queue);
            gpumem.deallocate(dev_X, (*N)*sizeof(magmaDoubleComplex));
   }
} // linalg

#endif

#endif
