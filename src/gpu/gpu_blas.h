#ifdef GPU

#ifndef GPU_BLAS_H
#define GPU_BLAS_H

#include <vector>
#include <complex>

#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

   // copy
   inline void xcopy_magma(const magma_int_t N, const double* X, double* Y){
      const magma_int_t INCX = 1, INCY = 1;
      magma_dcopy(N, X, INCX, Y, INCY, magma_queue);
   }
   inline void xcopy_magma(const magma_int_t N, const std::complex<double>* X, std::complex<double>* Y){
      const magma_int_t INCX = 1, INCY = 1;
      magma_zcopy(N, (magmaDoubleComplex *)X, INCX, (magmaDoubleComplex *)Y, INCY, magma_queue);
   }

   // axpy
   inline void xaxpy_magma(const magma_int_t N, const double alpha, 
         const double* X, double* Y){
      const magma_int_t INCX = 1, INCY = 1;
      magma_daxpy(N, alpha, X, INCX, Y, INCY, magma_queue);
   }
   inline void xaxpy_magma(const magma_int_t N, const std::complex<double> alpha, 
         const std::complex<double>* X, std::complex<double>* Y){
      const magma_int_t INCX = 1, INCY = 1;
      magmaDoubleComplex alpha1{alpha.real(),alpha.imag()};
      magma_zaxpy(N, alpha1, (magmaDoubleComplex *)X, INCX, (magmaDoubleComplex *)Y, INCY, magma_queue);
   }

   // y = alpha*A*x + beta*y
   inline void xgemv_magma(const char* TRANSA, const magma_int_t M, const magma_int_t N,
         const double alpha, const double* A, const magma_int_t LDA,
         const double* X, const magma_int_t INCX,
         const double beta, double* Y, const magma_int_t INCY){
      magma_trans_t transA =  MagmaNoTrans;
      if(TRANSA[0]=='T'){
         transA = MagmaTrans;
      }else if (TRANSA[0] == 'C'){
         transA = MagmaConjTrans;
      }
      size_t size = N*sizeof(double);
      double* dev_X = (double*)GPUmem.allocate(size);
      GPUmem.to_gpu(dev_X, X, size);
      magma_dgemv(transA, M, N, alpha, A, LDA,
            dev_X, INCX, beta, Y, INCY, magma_queue);
      GPUmem.deallocate(dev_X, size);
   }
   inline void xgemv_magma(const char* TRANSA, const magma_int_t M, const magma_int_t N,
         const std::complex<double> alpha, const std::complex<double>* A, const magma_int_t LDA,
         const std::complex<double>* X, const magma_int_t INCX,
         const std::complex<double> beta, std::complex<double>* Y, const magma_int_t INCY){
      magma_trans_t transA =  MagmaNoTrans;
      if(TRANSA[0] == 'T'){
         transA = MagmaTrans;
      }else if (TRANSA[0] == 'C'){
         transA = MagmaConjTrans;
      }
      size_t size = N*sizeof(magmaDoubleComplex);
      magmaDoubleComplex* dev_X = (magmaDoubleComplex*)GPUmem.allocate(size);
      GPUmem.to_gpu(dev_X, X, size);
      magmaDoubleComplex alpha1{alpha.real(),alpha.imag()};
      magmaDoubleComplex beta1{beta.real(),beta.imag()};
      magma_zgemv(transA, M, N, alpha1, (magmaDoubleComplex *)A, LDA,
            (magmaDoubleComplex *)dev_X, INCX, beta1, (magmaDoubleComplex *)Y, INCY, magma_queue);
      GPUmem.deallocate(dev_X, size);
   }
} // linalg

#endif

#endif
