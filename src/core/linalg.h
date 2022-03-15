#ifndef LINALG_H
#define LINALG_H

#include "blas.h"
#include "matrix.h"

extern "C" {
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

// wrapper for LAPACK
namespace linalg{

// eigendecomposition HU=Ue: order=0/1 small-large/large-small
void eig_solver(const matrix<double>& A, std::vector<double>& e, 
		matrix<double>& U, const int order=0);
void eig_solver(const matrix<std::complex<double>>& A, std::vector<double>& e, 
		matrix<std::complex<double>>& U, const int order=0);

// singular value decomposition: 
// iop =  0 : JOBU=A, JOBVT=A: all
//     =  1 : JOBU=S, JOBVT=N: the first min(m,n) columns of U 
//     =  2 : JOBU=N, JOBVT=S: the first min(m,n) rows of Vt
//     =  3 : JOBU=S, JOBVT=S: both
//     = 10 : JOBZ=A - divide-and-conquer version
//     = 13 : JOBZ=S (default)
void svd_solver(const matrix<double>& A, std::vector<double>& s, 
		matrix<double>& U, matrix<double>& Vt, 
		const int iop=13);
void svd_solver(const matrix<std::complex<double>>& A, std::vector<double>& s, 
		matrix<std::complex<double>>& U, matrix<std::complex<double>>& Vt, 
		const int iop=13);

} // linalg

#endif
