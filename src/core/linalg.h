#ifndef LINALG_H
#define LINALG_H

#include "blas.h"
#include "matrix.h"

extern "C" {
// eig
void dsyevd_(const char* JOBZ, const char* UPLO, const MKL_INT* N, 
	     double* A, const MKL_INT* LDA, 
	     double* W, double* WORK, const MKL_INT* LWORK, 
	     MKL_INT* IWORK, const MKL_INT* LIWORK, MKL_INT* INFO);
void zheevd_(const char* JOBZ, const char* UPLO, const MKL_INT* N, 
	     std::complex<double>* A, const MKL_INT* LDA, 
	     double* W, std::complex<double>* WORK, const MKL_INT* LWORK,
	     double* RWORK, const MKL_INT* LRWORK, 
	     MKL_INT* IWORK, const MKL_INT* LIWORK, MKL_INT* INFO);
// svd
void dgesvd_(const char* JOBU, const char* JOBVT, const MKL_INT* M, const MKL_INT* N,
	     double* A, const MKL_INT* LDA, double* S, 
	     double* U, const MKL_INT* LDU,
	     double* VT, const MKL_INT* LDVT, 
	     double* WORK, const MKL_INT* LWORK, MKL_INT* INFO);
void zgesvd_(const char* JOBU, const char* JOBVT, const MKL_INT* M, const MKL_INT* N,
	     std::complex<double>* A, const MKL_INT* LDA, double* S, 
	     std::complex<double>* U, const MKL_INT* LDU,
	     std::complex<double>* VT, const MKL_INT* LDVT, 
	     std::complex<double>* WORK, const MKL_INT* LWORK, 
	     double* RWORK, MKL_INT* INFO);
void dgesdd_(const char* JOBZ, const MKL_INT* M, const MKL_INT* N,
	     double* A, const MKL_INT* LDA, double* S, 
	     double* U, const MKL_INT* LDU,
	     double* VT, const MKL_INT* LDVT, 
	     double* WORK, const MKL_INT* LWORK, 
	     MKL_INT* IWORK, MKL_INT* INFO);
void zgesdd_(const char* JOBZ, const MKL_INT* M, const MKL_INT* N,
	     std::complex<double>* A, const MKL_INT* LDA, double* S, 
	     std::complex<double>* U, const MKL_INT* LDU,
	     std::complex<double>* VT, const MKL_INT* LDVT, 
	     std::complex<double>* WORK, const MKL_INT* LWORK, 
	     double* RWORK, MKL_INT* IWORK, MKL_INT* INFO);
}

// wrapper for LAPACK
namespace linalg{

// eigendecomposition HU=Ue: order=0/1 small-large/large-small
void eig_solver(const matrix<double>& A, std::vector<double>& e, 
		matrix<double>& U, const MKL_INT order=0);
void eig_solver(const matrix<std::complex<double>>& A, std::vector<double>& e, 
		matrix<std::complex<double>>& U, const MKL_INT order=0);

// singular value decomposition: 
// iop =  0 : JOBU=A, JOBVT=A: all
//     =  1 : JOBU=S, JOBVT=N: the first min(m,n) columns of U 
//     =  2 : JOBU=N, JOBVT=S: the first min(m,n) rows of Vt
//     =  3 : JOBU=S, JOBVT=S: both
//     = 10 : JOBZ=A - divide-and-conquer version
//     = 13 : JOBZ=S (default)
void svd_solver(const matrix<double>& A, std::vector<double>& s, 
		matrix<double>& U, matrix<double>& Vt, 
		const MKL_INT iop=13);
void svd_solver(const matrix<std::complex<double>>& A, std::vector<double>& s, 
		matrix<std::complex<double>>& U, matrix<std::complex<double>>& Vt, 
		const MKL_INT iop=13);

} // linalg

#endif
