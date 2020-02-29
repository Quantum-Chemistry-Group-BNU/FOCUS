#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include "matrix.h"

extern "C" {

// dimension should be const 	
void dgemm_(const char* TRANSA, const char* TRANSB, const int* M, const int* N, const int* K,
	    const double* alpha, double* A, const int* LDA, double* B, const int* LDB,
            const double* beta, double* C, const int* LDC);

void dsyevd_(const char* JOBZ, const char* UPLO, const int* N, double* A, const int* LDA, 
	     double* W, double* WORK, const int* LWORK, 
	     int* IWORK, const int* LIWORK, int* INFO);

double dnrm2_(const int* N, double* X, const int* INCX);

}

// wrapper for BLAS/LAPACK with matrix class
namespace linalg{

// C = alpha*A*B + beta*C
void dgemm(const char* TRANSA, const char* TRANSB,
	   const double alpha, const matrix& A, const matrix& B,
	   const double beta, matrix& C);

// eigenvalues
void eig(matrix& A, std::vector<double>& e);

// Fnorm
double Fnorm(const matrix& A);

} // linalg

#endif
