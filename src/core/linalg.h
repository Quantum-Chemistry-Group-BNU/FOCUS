#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include "matrix.h"

extern "C" {
// nrm2
double dnrm2_(const int* N, const double* X, const int* INCX);
double dznrm2_(const int* N, const std::complex<double>* X, const int* INCX);
// dot
double ddot_(const int* N, const double* X, const int* INCX,
	     const double* Y, const int* INCY);
std::complex<double> zdotc_(const int* N, const std::complex<double>* ZX, const int* INCX,
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
// eig
void dsyevd_(const char* JOBZ, const char* UPLO, const int* N, 
	     double* A, const int* LDA, 
	     double* W, double* WORK, const int* LWORK, 
	     int* IWORK, const int* LIWORK, int* INFO);
void zheevd_(const char* JOBZ, const char* UPLO, const int* N, 
	     std::complex<double>* A, const int* LDA, 
	     double* W, std::complex<double>* WORK, const int* LWORK,
	     double* RWORK, const int* LRWORK, 
	     int* IWORK, const int* LIWORK, int* INFO);
// svd
void dgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N,
	     double* A, const int* LDA, double* S, 
	     double* U, const int* LDU,
	     double* VT, const int* LDVT, 
	     double* WORK, const int* LWORK, int* INFO);
void zgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N,
	     std::complex<double>* A, const int* LDA, double* S, 
	     std::complex<double>* U, const int* LDU,
	     std::complex<double>* VT, const int* LDVT, 
	     std::complex<double>* WORK, const int* LWORK, 
	     double* RWORK, int* INFO);
void dgesdd_(const char* JOBZ, const int* M, const int* N,
	     double* A, const int* LDA, double* S, 
	     double* U, const int* LDU,
	     double* VT, const int* LDVT, 
	     double* WORK, const int* LWORK, 
	     int* IWORK, int* INFO);
void zgesdd_(const char* JOBZ, const int* M, const int* N,
	     std::complex<double>* A, const int* LDA, double* S, 
	     std::complex<double>* U, const int* LDU,
	     std::complex<double>* VT, const int* LDVT, 
	     std::complex<double>* WORK, const int* LWORK, 
	     double* RWORK, int* IWORK, int* INFO);
}

// wrapper for BLAS/LAPACK with matrix class
namespace linalg{

// xnrm2
inline double xnrm2(const int N, const double* X){
   int INCX = 1;
   return ::dnrm2_(&N, X, &INCX);
}
inline double xnrm2(const int N, const std::complex<double>* X){
   int INCX = 1;
   return ::dznrm2_(&N, X, &INCX);
}

// xdot
inline double xdot(const int N, const double* X, const double* Y){
   int INCX = 1, INCY = 1;
   return ::ddot_(&N, X, &INCX, Y, &INCY);
}
inline std::complex<double> xdot(const int N, const std::complex<double>* X, const std::complex<double>* Y){
   int INCX = 1, INCY = 1;
   return ::zdotc_(&N, X, &INCX, Y, &INCY);
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
// shorthand for A*B
template <typename Tm>
matrix<Tm> xgemm(const char* TRANSA, const char* TRANSB,
	         const matrix<Tm>& A, const matrix<Tm>& B,
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
   matrix<Tm> C(M,N);
   LDC = M; // we assume the exact match of matrix size in our applications
   const Tm beta = 0.0;
   xgemm(&trans_A, &trans_B, &M, &N, &K, &alpha, 
	 A.data(), &LDA, B.data(), &LDB, &beta, 
	 C.data(), &LDC);
   return C;
}

// eigendecomposition HU=Ue: order=0/1 small-large/large-small
void eig_solver(const matrix<double>& A, std::vector<double>& e, 
		matrix<double>& U, const int order=0);
void eig_solver(const matrix<std::complex<double>>& A, std::vector<double>& e, 
		matrix<std::complex<double>>& U, const int order=0);

// singular value decomposition: 
// iop =  0 : JOBU=A, JOBVT=A
//     =  1 : JOBU=S, JOBVT=N
//     =  2 : JOBU=N, JOBVT=S
//     =  3 : JOBU=S, JOBVT=S 
//     = 10 : JOBZ=A - divide-and-conquer version
//     = 13 : JOBZ=S
void svd_solver(const matrix<double>& A, std::vector<double>& s, 
		matrix<double>& U, matrix<double>& Vt, 
		const int iop=1);
void svd_solver(const matrix<std::complex<double>>& A, std::vector<double>& s, 
		matrix<std::complex<double>>& U, matrix<std::complex<double>>& Vt, 
		const int iop=1);

} // linalg

#endif
