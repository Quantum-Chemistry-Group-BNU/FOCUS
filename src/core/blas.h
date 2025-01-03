#ifndef BLAS_H
#define BLAS_H

#include <vector>
#include <complex>
//xiangchunyang 20250103
extern "C" {
#include <cblas_optimized.h>
}

// This is consistent with magma_int_t in magma_types.h
#ifdef MKL_ILP64
   #define MKL_INT long long int
#else
   #define MKL_INT int
#endif
/*
#if defined(MAGMA_ILP64) || defined(MKL_ILP64)
typedef long long int magma_int_t;  // MKL uses long long int, not int64_t
#else
typedef int magma_int_t;
#endif
*/

extern "C" {
   // conj:
   void zlacgv_(const MKL_INT* N, const std::complex<double>* X, const MKL_INT* INCX);
   // scal: x = a*x
   void dscal_(const MKL_INT* N, const double* alpha, double* X, const MKL_INT* INCX);
   void zscal_(const MKL_INT* N, const std::complex<double>* alpha, std::complex<double>* X, const MKL_INT* INCX);
   // copy: y = x
   void dcopy_(const MKL_INT* N,
         const double* X, const MKL_INT* INCX,
         double* Y, const MKL_INT* INCY);
   void zcopy_(const MKL_INT* N,
         const std::complex<double>* X, const MKL_INT* INCX,
         std::complex<double>* Y, const MKL_INT* INCY);
   // axpy: y += a*x
   void daxpy_(const MKL_INT* N, const double* alpha, 
         const double* X, const MKL_INT* INCX,
         double* Y, const MKL_INT* INCY);
   void zaxpy_(const MKL_INT* N, const std::complex<double>* alpha,
         const std::complex<double>* X, const MKL_INT* INCX,
         std::complex<double>* Y, const MKL_INT* INCY);
   // nrm2
   double dnrm2_(const MKL_INT* N, const double* X, const MKL_INT* INCX);
   double dznrm2_(const MKL_INT* N, const std::complex<double>* X, const MKL_INT* INCX);
   // dot
   double ddot_(const MKL_INT* N, const double* X, const MKL_INT* INCX,
         const double* Y, const MKL_INT* INCY);
   void zdotc_(std::complex<double>* result, 
         const MKL_INT* N, const std::complex<double>* ZX, const MKL_INT* INCX,
         const std::complex<double>* ZY, const MKL_INT* INCY); // X^H * Y
   // gemm	
   void dgemm_(const char* TRANSA, const char* TRANSB, 
         const MKL_INT* M, const MKL_INT* N, const MKL_INT* K,
         const double* alpha, const double* A, const MKL_INT* LDA, 
         const double* B, const MKL_INT* LDB,
         const double* beta, double* C, const MKL_INT* LDC);
   void zgemm_(const char* TRANSA, const char* TRANSB, 
         const MKL_INT* M, const MKL_INT* N, const MKL_INT* K,
         const std::complex<double>* alpha, const std::complex<double>* A, const MKL_INT* LDA, 
         const std::complex<double>* B, const MKL_INT* LDB,
         const std::complex<double>* beta, std::complex<double>* C, const MKL_INT* LDC);
   // gemv
   void dgemv_(const char* TRANSA, const MKL_INT* M, const MKL_INT* N, 
         const double* alpha, const double* A, const MKL_INT* LDA,
         const double* X, const MKL_INT* INCX,
         const double* beta, double* Y, const MKL_INT* INCY);
   // zgemv
   void zgemv_(const char* TRANSA, const MKL_INT* M, const MKL_INT* N, 
         const std::complex<double>* alpha, const std::complex<double>* A, const MKL_INT* LDA,
         const std::complex<double>* X, const MKL_INT* INCX,
         const std::complex<double>* beta, std::complex<double>* Y, const MKL_INT* INCY);
}
double dnrm2_optimized(const double* x, const MKL_INT n);

// wrapper for BLAS
namespace linalg{

   // conj
   inline void xconj(const MKL_INT N, double* X){
      return;
   }
   inline void xconj(const MKL_INT N, std::complex<double>* X){
      MKL_INT INCX = 1;
      ::zlacgv_(&N, X, &INCX);
   }

   // scal
   inline void xscal(const MKL_INT N, const double alpha, double* X){
      MKL_INT INCX = 1;
      ::dscal_(&N, &alpha, X, &INCX);
   }
   inline void xscal(const MKL_INT N, const std::complex<double> alpha, std::complex<double>* X){
      MKL_INT INCX = 1;
      ::zscal_(&N, &alpha, X, &INCX);
   }

   // copy
   inline void xcopy(const MKL_INT N, const double* X, double* Y){
      MKL_INT INCX = 1, INCY = 1;
      //::dcopy_(&N, X, &INCX, Y, &INCY);
      cblas_dcopy_optimized((MKL_INT)N, (double*)X, INCX, Y, INCY);
   }
   inline void xcopy(const MKL_INT N, const std::complex<double>* X, std::complex<double>* Y){
      MKL_INT INCX = 1, INCY = 1;
      //::zcopy_(&N, X, &INCX, Y, &INCY);
      cblas_zcopy_optimized((MKL_INT)N, (void*)X, INCX, Y, INCY);
   }

   // general copy with offset
   inline void xcopy(const MKL_INT N, const double* X, const MKL_INT INCX, 
         double* Y, const MKL_INT INCY){
      ::dcopy_(&N, X, &INCX, Y, &INCY);
   }
   inline void xcopy(const MKL_INT N, const std::complex<double>* X, const MKL_INT INCX, 
         std::complex<double>* Y, const MKL_INT INCY){
      ::zcopy_(&N, X, &INCX, Y, &INCY);
   }

   // xaxpy
   inline void xaxpy(const MKL_INT N, const double alpha, 
         const double* X, double* Y){
      MKL_INT INCX = 1, INCY = 1;
      ::daxpy_(&N, &alpha, X, &INCX, Y, &INCY);
   }
   inline void xaxpy(const MKL_INT N, const std::complex<double> alpha, 
         const std::complex<double>* X, std::complex<double>* Y){
      MKL_INT INCX = 1, INCY = 1;
      ::zaxpy_(&N, &alpha, X, &INCX, Y, &INCY);
   }

   // xnrm2
   inline double xnrm2(const MKL_INT N, const double* X){
      MKL_INT INCX = 1;
      //return ::dnrm2_(&N, X, &INCX);
      return cblas_dnrm2_optimized((MKL_INT)N, (double*)X, INCX);
   }
   inline double xnrm2(const MKL_INT N, const std::complex<double>* X){
      MKL_INT INCX = 1;
      //return ::dznrm2_(&N, X, &INCX);
      return cblas_znrm2_optimized((MKL_INT)N, (void*)X, INCX);
   }

   // xdot
   inline double xdot(const MKL_INT N, const double* X, const double* Y){
      MKL_INT INCX = 1, INCY = 1;
      return ::ddot_(&N, X, &INCX, Y, &INCY);
   }
   inline std::complex<double> xdot(const MKL_INT N, const std::complex<double>* X, const std::complex<double>* Y){
      MKL_INT INCX = 1, INCY = 1;
      // ZL@20200622: This works for MKL, while others may return complex value directly
      std::complex<double> result;
      ::zdotc_(&result, &N, X, &INCX, Y, &INCY);
      return result;
   }

   // C = alpha*A*B + beta*C
   inline void xgemm(const char* TRANSA, const char* TRANSB,
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const double alpha, const double* A, const MKL_INT LDA, 
         const double* B, const MKL_INT LDB,
         const double beta, double* C, const MKL_INT LDC){
      ::dgemm_(TRANSA, TRANSB, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
   }
   inline void xgemm(const char* TRANSA, const char* TRANSB, 
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const std::complex<double> alpha, const std::complex<double>* A, const MKL_INT LDA, 
         const std::complex<double>* B, const MKL_INT LDB,
         const std::complex<double> beta, std::complex<double>* C, const MKL_INT LDC){
      ::zgemm_(TRANSA, TRANSB, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
   }
   // C = alpha*A*B + beta*C
   inline void xgemm_small(const char* TRANSA, const char* TRANSB,
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const double alpha, const double* A, const MKL_INT LDA, 
         const double* B, const MKL_INT LDB,
         const double beta, double* C, const MKL_INT LDC){
      //::dgemm_(TRANSA, TRANSB, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
       CBLAS_TRANSPOSE transa_, transb_;
      if(*TRANSA=='N') 
      {
          transa_=CblasNoTrans;
      }else if(*TRANSA =='T')
      {
          transa_=CblasTrans;
      }else if(*TRANSA == 'R')
      {
          transa_=CblasConjNoTrans;
      }else{
          transa_=CblasConjTrans;
      }

      if(*TRANSB=='N') 
      {
          transb_=CblasNoTrans;
      }else if(*TRANSB =='T')
      {
          transb_=CblasTrans;
      }else if(*TRANSB == 'R')
      {
          transb_=CblasConjNoTrans;
      }else{
          transb_=CblasConjTrans;
      }

      cblas_dgemm_small(CblasColMajor, transa_, transb_, M, N, K, alpha, (double*)A, LDA, (double*)B, LDB, beta, (double*)C, LDC);
   }
   inline void xgemm_small(const char* TRANSA, const char* TRANSB, 
         const MKL_INT M, const MKL_INT N, const MKL_INT K,
         const std::complex<double> alpha, const std::complex<double>* A, const MKL_INT LDA, 
         const std::complex<double>* B, const MKL_INT LDB,
         const std::complex<double> beta, std::complex<double>* C, const MKL_INT LDC){
      //::zgemm_(TRANSA, TRANSB, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
      enum CBLAS_TRANSPOSE transa_, transb_;
      if(*TRANSA=='N') 
      {
          transa_=CblasNoTrans;
      }else if(*TRANSA =='T')
      {
          transa_=CblasTrans;
      }else if(*TRANSA == 'R')
      {
          transa_=CblasConjNoTrans;
      }else{
          transa_=CblasConjTrans;
      }

      if(*TRANSB=='N') 
      {
          transb_=CblasNoTrans;
      }else if(*TRANSB =='T')
      {
          transb_=CblasTrans;
      }else if(*TRANSB == 'R')
      {
          transb_=CblasConjNoTrans;
      }else{
          transb_=CblasConjTrans;
      }

      cblas_zgemm_small(CblasColMajor, transa_, transb_, M, N, K, (void*)(&alpha), (void*)A, LDA, (void*)B, LDB, (void*)(&beta), (void*)C, LDC);
   }

   // y = alpha*A*x + beta*y
   inline void xgemv(const char* TRANSA, const MKL_INT M, const MKL_INT N, 
         const double alpha, const double* A, const MKL_INT LDA,
         const double* X, const MKL_INT INCX,
         const double beta, double* Y, const MKL_INT INCY){
      ::dgemv_(TRANSA, &M, &N, &alpha, A, &LDA, X, &INCX, &beta, Y, &INCY);
   }
   // zgemv
   inline void xgemv(const char* TRANSA, const MKL_INT M, const MKL_INT N, 
         const std::complex<double> alpha, const std::complex<double>* A, const MKL_INT LDA,
         const std::complex<double>* X, const MKL_INT INCX,
         const std::complex<double> beta, std::complex<double>* Y, const MKL_INT INCY){
      ::zgemv_(TRANSA, &M, &N, &alpha, A, &LDA, X, &INCX, &beta, Y, &INCY);
   }

} // linalg

#endif
