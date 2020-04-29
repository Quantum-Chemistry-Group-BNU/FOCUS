#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include "matrix.h"

extern "C" {

// dimension should be const, A & B do not change	
void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
	    const double* alpha, const double* A, const int* LDA, const double* B, const int* LDB,
            const double* beta, double* C, const int* LDC);

void dsyevd_(const char* JOBZ, const char* UPLO, const int* N, double* A, const int* LDA, 
	     double* W, double* WORK, const int* LWORK, 
	     int* IWORK, const int* LIWORK, int* INFO);

void dgesvd_(const char* JOBU, const char* JOBVT, const int* M, const int* N,
	     double* A, const int* LDA, double* S, double* U, const int* LDU,
	     double* VT, const int* LDVT, double* WORK, const int* LWORK, int* INFO);

void dgesdd_(const char* JOBZ, const int* M, const int* N,
	     double* A, const int* LDA, double* S, double* U, const int* LDU,
	     double* VT, const int* LDVT, double* WORK, const int* LWORK, 
	     int* IWORK, int* INFO);
	
double dnrm2_(const int* N, const double* X, const int* INCX);

double ddot_(const int* N, const double* X, const int* INCX,
	     const double* Y, const int* INCY);

}

// wrapper for BLAS/LAPACK with matrix class
namespace linalg{

// C = alpha*A*B + beta*C
void dgemm(const char* TRANSA, const char* TRANSB,
	   const double alpha, const matrix& A, const matrix& B,
	   const double beta, matrix& C);

// shorthand for A*B
matrix dgemm(const char* TRANSA, const char* TRANSB,
	     const matrix& A, const matrix& B,
	     const double alpha=1.0);

// eigenvalues: 
// order=0 from small to large; 
// order=1 from large to small.
void eigen_solver(matrix& A, std::vector<double>& e, const int order=0);

// singular value decomposition
void svd_solver(matrix& A, std::vector<double>& s, 
		matrix& U, matrix& Vt, 
		const int iop=0);

// normF = ||A||_F = sqrt(\sum_{ij}|aij|^2)
double normF(const matrix& A);

// nrm2
double dnrm2(const int N, const double* X);

// ddot2
double ddot(const int N, const double* X, const double* Y);

// ||A - At||_F
double symmetric_diff(const matrix& A);

} // linalg

#endif
