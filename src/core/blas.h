#ifndef BLAS_H
#define BLAS_H

#include <vector>
#include <complex>
#include "matrix.h"

extern "C" {
// conj:
void zlacgv_(const int* N, const std::complex<double>* X, const int* INCX);
// scal: x = a*x
void dscal_(const int* N, const double* alpha, double* X, const int* INCX);
void zscal_(const int* N, const std::complex<double>* alpha, std::complex<double>* X, const int* INCX);
// copy: y = x
void dcopy_(const int* N,
	    const double* X, const int* INCX,
	    double* Y, const int* INCY);
void zcopy_(const int* N,
	    const std::complex<double>* X, const int* INCX,
	    std::complex<double>* Y, const int* INCY);
// axpy: y += a*x
void daxpy_(const int* N, const double* alpha, 
	    const double* X, const int* INCX,
	    double* Y, const int* INCY);
void zaxpy_(const int* N, const std::complex<double>* alpha,
	    const std::complex<double>* X, const int* INCX,
	    std::complex<double>* Y, const int* INCY);
// nrm2
double dnrm2_(const int* N, const double* X, const int* INCX);
double dznrm2_(const int* N, const std::complex<double>* X, const int* INCX);
// dot
double ddot_(const int* N, const double* X, const int* INCX,
	     const double* Y, const int* INCY);
void zdotc_(std::complex<double>* result, 
	    const int* N, const std::complex<double>* ZX, const int* INCX,
	    const std::complex<double>* ZY, const int* INCY); // X^H * Y
// gemm	
void dgemm_(const char* TRANSA, const char* TRANSB, 
	    const int* M, const int* N, const int* K,
	    const double* alpha, const double* A, const int* LDA, 
	    const double* B, const int* LDB,
            const double* beta, double* C, const int* LDC);
void zgemm_(const char* TRANSA, const char* TRANSB, 
	    const int* M, const int* N, const int* K,
	    const std::complex<double>* alpha, const std::complex<double>* A, const int* LDA, 
	    const std::complex<double>* B, const int* LDB,
            const std::complex<double>* beta, std::complex<double>* C, const int* LDC);
}

// wrapper for BLAS
namespace linalg{

// conj
inline void xconj(const int N, const double* X){
   int INCX = 1;
   ::zlacgv_(&N, X, &INCX);
}

// scal
inline void xscal(const int N, const double alpha, double* X){
   int INCX = 1;
   ::dscal_(&N, &alpha, X, &INCX);
}
inline void xscal(const int N, const std::complex<double> alpha, std::complex<double>* X){
   int INCX = 1;
   ::zscal_(&N, &alpha, X, &INCX);
}

// copy
inline void xcopy(const int N, const double* X, double* Y){
   int INCX = 1, INCY = 1;
   ::dcopy_(&N, X, &INCX, Y, &INCY);
}
inline void xcopy(const int N, const std::complex<double>* X, std::complex<double>* Y){
   int INCX = 1, INCY = 1;
   ::zcopy_(&N, X, &INCX, Y, &INCY);
}

// xaxpy
inline void xaxpy(const int N, const double alpha, 
		  const double* X, double* Y){
   int INCX = 1, INCY = 1;
   ::daxpy_(&N, &alpha, X, &INCX, Y, &INCY);
}
inline void xaxpy(const int N, const std::complex<double> alpha, 
		  const std::complex<double>* X, std::complex<double>* Y){
   int INCX = 1, INCY = 1;
   ::zaxpy_(&N, &alpha, X, &INCX, Y, &INCY);
}

// xnrm2
inline double xnrm2(const int N, const double* X){
   int INCX = 1;
   return ::dnrm2_(&N, X, &INCX);
}
inline double xnrm2(const int N, const std::complex<double>* X){
   int INCX = 1;
   return ::dznrm2_(&N, X, &INCX);
}

// normF = ||A||_F = sqrt(\sum_{ij}|aij|^2)
template <typename Tm>
double normF(const matrix<Tm>& A){
   return xnrm2(A.size(), A.data());
}

// ||A - Ah||_F
template <typename Tm>
double symmetric_diff(const matrix<Tm>& A){
   assert(A.rows() == A.cols());
   return normF(A-A.H());
}

// xdot
inline double xdot(const int N, const double* X, const double* Y){
   int INCX = 1, INCY = 1;
   return ::ddot_(&N, X, &INCX, Y, &INCY);
}
inline std::complex<double> xdot(const int N, const std::complex<double>* X, const std::complex<double>* Y){
   int INCX = 1, INCY = 1;
   // ZL@20200622: This works for MKL, while others may return complex value directly
   std::complex<double> result;
   ::zdotc_(&result, &N, X, &INCX, Y, &INCY);
   return result;
}

// C = alpha*A*B + beta*C
inline void xgemm(const char* TRANSA, const char* TRANSB,
	          const int* M, const int* N, const int* K,
	          const double* alpha, const double* A, const int* LDA, 
	          const double* B, const int* LDB,
                  const double* beta, double* C, const int* LDC){
   return ::dgemm_(TRANSA, TRANSB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
}
inline void xgemm(const char* TRANSA, const char* TRANSB, 
	          const int* M, const int* N, const int* K,
	          const std::complex<double>* alpha, const std::complex<double>* A, const int* LDA, 
	          const std::complex<double>* B, const int* LDB,
                  const std::complex<double>* beta, std::complex<double>* C, const int* LDC){
   return ::zgemm_(TRANSA, TRANSB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);
}
// C := alpha*op( A )*op( B )
template <typename Tm>
matrix<Tm> xgemm(const char* TRANSA, const char* TRANSB,
	         const BaseMatrix<Tm>& A, const BaseMatrix<Tm>& B,
		 const Tm alpha=1.0){
   int M, N, K;
   int LDA, LDB, LDC;
   // TRANS is c-type string (character array), input "N" not 'N' 
   char trans_A = toupper(TRANSA[0]); 
   char trans_B = toupper(TRANSB[0]);
   assert(trans_A == 'N' || trans_A == 'T' || trans_A == 'C');
   assert(trans_B == 'N' || trans_B == 'T' || trans_B == 'C');
   if(trans_A == 'N'){
      M = A.rows(); K = A.cols(); LDA = M; 
   }else{
      M = A.cols(); K = A.rows(); LDA = K; 
   }
   if(trans_B == 'N'){
      assert(K == B.rows());
      N = B.cols(); LDB = K;
   }else{
      assert(K == B.cols());
      N = B.rows(); LDB = N;
   }
   // we assume the exact match of matrix size in our applications
   matrix<Tm> C(M,N);
   LDC = M;
   const Tm beta = 0.0;
   xgemm(&trans_A, &trans_B, &M, &N, &K, &alpha, 
	 A.data(), &LDA, B.data(), &LDB, &beta, 
	 C.data(), &LDC);
   return C;
}

} // linalg

#endif
