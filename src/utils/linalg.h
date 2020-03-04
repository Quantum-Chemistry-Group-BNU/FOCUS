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
matrix mdot(const matrix& A, const matrix& B);

// eigenvalues
void eigen_solver(matrix& A, std::vector<double>& e);

// normF
double normF(const matrix& A);

// nrm2
double dnrm2(const int N, const double* X);

// ddot2
double ddot(const int N, const double* X, const double* Y);

// transpose
matrix transpose(const matrix& A);

// ||A - At||F
double symmetric_diff(const matrix& A);

} // linalg

#endif
