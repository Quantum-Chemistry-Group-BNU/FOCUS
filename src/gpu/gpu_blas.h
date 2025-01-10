#ifdef GPU

#ifndef GPU_BLAS_H
#define GPU_BLAS_H

#include <vector>
#include <complex>
#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

   // dcopy
   inline void xcopy_gpu(const MKL_INT N, const double* X, double* Y, const int iop=0){
      if(iop == 0){
         assert(N <= INT_MAX);
         CUBLAS_CHECK(cublasDcopy(handle_cublas, N, X, 1, Y, 1));
#ifdef MAGMA
      }else{
         magma_dcopy(N, X, 1, Y, 1, magma_queue);
#else  
      }else{
         std::cout << "error: no such option in xcopy_gpu: iop=" << iop << std::endl;
         exit(1);
#endif
      }
   }

   // zcopy
   inline void xcopy_gpu(const MKL_INT N, const std::complex<double>* X, std::complex<double>* Y, const int iop=0){
      if(iop == 0){
         assert(N <= INT_MAX);
         CUBLAS_CHECK(cublasZcopy(handle_cublas, N, (cuDoubleComplex *)X, 1, (cuDoubleComplex *)Y, 1));
#ifdef MAGMA
      }else{
         magma_zcopy(N, (magmaDoubleComplex *)X, 1, (magmaDoubleComplex *)Y, 1, magma_queue);
#else  
      }else{
         std::cout << "error: no such option in xcopy_gpu: iop=" << iop << std::endl;
         exit(1);
#endif
      }
   }

   // daxpy
   inline void xaxpy_gpu(const MKL_INT N, const double alpha, 
         const double* X, double* Y, const int iop=0){
      if(iop == 0){
         assert(N <= INT_MAX);
         CUBLAS_CHECK(cublasDaxpy(handle_cublas, N, &alpha, X, 1, Y, 1));
#ifdef MAGMA
      }else{
         magma_daxpy(N, alpha, X, 1, Y, 1, magma_queue);
#else  
      }else{
         std::cout << "error: no such option in xaxpy_gpu: iop=" << iop << std::endl;
         exit(1);
#endif
      }
   }
   // zaxpy
   inline void xaxpy_gpu(const MKL_INT N, const std::complex<double> alpha, 
         const std::complex<double>* X, std::complex<double>* Y, const int iop=0){
      if(iop == 0){
         assert(N <= INT_MAX);
         CUBLAS_CHECK(cublasZaxpy(handle_cublas, N, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)X, 1, (cuDoubleComplex *)Y, 1));
#ifdef MAGMA
      }else{
         magmaDoubleComplex alpha1{alpha.real(),alpha.imag()};
         magma_zaxpy(N, alpha1, (magmaDoubleComplex *)X, 1, (magmaDoubleComplex *)Y, 1, magma_queue);
#else
      }else{
         std::cout << "error: no such option in xaxpy_gpu: iop=" << iop << std::endl;
         exit(1);
#endif
      }
   }

   inline void xgemm_gpu(const char* TRANSA, const char* TRANSB,
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const double alpha, const double* A, const MKL_INT LDA, 
         const double* B, const MKL_INT LDB,
         const double beta, double* C, const MKL_INT LDC){
      cublasOperation_t trans_a = (TRANSA[0]=='T' || TRANSA[0]=='C')? CUBLAS_OP_T : CUBLAS_OP_N;
      cublasOperation_t trans_b = (TRANSB[0]=='T' || TRANSB[0]=='C')? CUBLAS_OP_T : CUBLAS_OP_N;
      CUBLAS_CHECK(cublasDgemm(handle_cublas, trans_a, trans_b,
              M, N, K, &alpha, A, LDA, B, LDB, &beta, C, LDA)); 
   }
   inline void xgemm_gpu(const char* TRANSA, const char* TRANSB, 
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const std::complex<double> alpha, const std::complex<double>* A, const MKL_INT LDA, 
         const std::complex<double>* B, const MKL_INT LDB,
         const std::complex<double> beta, std::complex<double>* C, const MKL_INT LDC){
      cublasOperation_t trans_a = CUBLAS_OP_N;
      if(TRANSA[0] == 'T'){
         trans_a = CUBLAS_OP_T;
      }else if(TRANSA[0] == 'C'){
         trans_a = CUBLAS_OP_C;
      }
      cublasOperation_t trans_b = CUBLAS_OP_N;
      if(TRANSB[0] == 'T'){
         trans_b = CUBLAS_OP_T;
      }else if(TRANSB[0] == 'C'){
         trans_b = CUBLAS_OP_C;
      }
      CUBLAS_CHECK(cublasZgemm(handle_cublas, trans_a, trans_b,
              M, N, K, (cuDoubleComplex *)&alpha, (cuDoubleComplex *)A, LDA, 
              (cuDoubleComplex *)B, LDB, (cuDoubleComplex *)&beta, (cuDoubleComplex *)C, LDA)); 
   }

/*
   // y = alpha*A*x + beta*y
   inline void xgemv_magma(const char* TRANSA, const MKL_INT M, const MKL_INT N,
         const double alpha, const double* A, const MKL_INT LDA,
         const double* X, const MKL_INT INCX,
         const double beta, double* Y, const MKL_INT INCY){
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
   inline void xgemv_magma(const char* TRANSA, const MKL_INT M, const MKL_INT N,
         const std::complex<double> alpha, const std::complex<double>* A, const MKL_INT LDA,
         const std::complex<double>* X, const MKL_INT INCX,
         const std::complex<double> beta, std::complex<double>* Y, const MKL_INT INCY){
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
*/

} // linalg

#endif

#endif
