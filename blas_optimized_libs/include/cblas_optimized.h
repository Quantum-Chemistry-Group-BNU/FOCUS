#ifndef __CBLAS_OPTIMIZED_H__
#define __CBLAS_OPTIMIZED_H__
#include <common.h>

# define OPENBLAS_CONST      /* see comment in cblas.h */


typedef enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102} CBLAS_ORDER;
typedef enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, CblasConjNoTrans=114} CBLAS_TRANSPOSE;
typedef enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122} CBLAS_UPLO;
typedef enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132} CBLAS_DIAG;
typedef enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142} CBLAS_SIDE;
typedef CBLAS_ORDER CBLAS_LAYOUT;



void cblas_sgemm_small(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST float alpha, 
		OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST float beta, 
		float *C, OPENBLAS_CONST blasint ldc);
void cblas_dgemm_small(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST double alpha, 
		OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST double beta, 
		double *C, OPENBLAS_CONST blasint ldc);
void cblas_cgemm_small(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST void *alpha, 
		OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST void *beta, 
		void *C, OPENBLAS_CONST blasint ldc);
void cblas_zgemm_small(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST void *alpha, 
		OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST void *beta, 
		void *C, OPENBLAS_CONST blasint ldc);

void cblas_sgemm_optimized(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST float alpha, 
		OPENBLAS_CONST float *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST float *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST float beta, 
		float *C, OPENBLAS_CONST blasint ldc);
void cblas_dgemm_optimized(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST double alpha, 
		OPENBLAS_CONST double *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST double *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST double beta, 
		double *C, OPENBLAS_CONST blasint ldc);
void cblas_cgemm_optimized(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST void *alpha, 
		OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST void *beta, 
		void *C, OPENBLAS_CONST blasint ldc);
void cblas_zgemm_optimized(OPENBLAS_CONST enum CBLAS_ORDER Order, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA, OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB, 
		OPENBLAS_CONST blasint M, OPENBLAS_CONST blasint N, OPENBLAS_CONST blasint K,
		OPENBLAS_CONST void *alpha, 
		OPENBLAS_CONST void *A, OPENBLAS_CONST blasint lda, 
		OPENBLAS_CONST void *B, OPENBLAS_CONST blasint ldb, 
		OPENBLAS_CONST void *beta, 
		void *C, OPENBLAS_CONST blasint ldc);

void cblas_scopy_optimized(OPENBLAS_CONST blasint n, OPENBLAS_CONST float *x, OPENBLAS_CONST blasint incx, float *y, OPENBLAS_CONST blasint incy);
void cblas_dcopy_optimized(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx, double *y, OPENBLAS_CONST blasint incy);
void cblas_ccopy_optimized(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);
void cblas_zcopy_optimized(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, void *y, OPENBLAS_CONST blasint incy);

float  cblas_snrm2_optimized (OPENBLAS_CONST blasint N, OPENBLAS_CONST float  *X, OPENBLAS_CONST blasint incX);
double cblas_dnrm2_optimized (OPENBLAS_CONST blasint N, OPENBLAS_CONST double *X, OPENBLAS_CONST blasint incX);
float  cblas_cnrm2_optimized(OPENBLAS_CONST blasint N, OPENBLAS_CONST void  *X, OPENBLAS_CONST blasint incX);
double cblas_znrm2_optimized(OPENBLAS_CONST blasint N, OPENBLAS_CONST void *X, OPENBLAS_CONST blasint incX);

void dprecondition_optimized(long ndim, const double* rvec, double* tvec, const double ei, const double damping, const double* Diag);

#endif
