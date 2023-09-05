#ifndef __COMMON_LEVEL3_H__
#define __COMMON_LEVEL3_H__
int sgemm_small_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float beta, float * C, BLASLONG ldc);
int sgemm_small_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float beta, float * C, BLASLONG ldc);
int sgemm_small_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float beta, float * C, BLASLONG ldc);
int sgemm_small_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float beta, float * C, BLASLONG ldc);

int dgemm_small_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double beta, double * C, BLASLONG ldc);
int dgemm_small_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double beta, double * C, BLASLONG ldc);
int dgemm_small_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double beta, double * C, BLASLONG ldc);
int dgemm_small_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double beta, double * C, BLASLONG ldc);

int sgemm_small_kernel_b0_nn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float * C, BLASLONG ldc);
int sgemm_small_kernel_b0_nt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float * C, BLASLONG ldc);
int sgemm_small_kernel_b0_tn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float * C, BLASLONG ldc);
int sgemm_small_kernel_b0_tt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha, float * B, BLASLONG ldb, float * C, BLASLONG ldc);

int dgemm_small_kernel_b0_nn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double * C, BLASLONG ldc);
int dgemm_small_kernel_b0_nt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double * C, BLASLONG ldc);
int dgemm_small_kernel_b0_tn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double * C, BLASLONG ldc);
int dgemm_small_kernel_b0_tt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha, double * B, BLASLONG ldb, double * C, BLASLONG ldc);

int cgemm_small_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_nr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_nc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
	
int cgemm_small_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_tr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_tc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);

int cgemm_small_kernel_rn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_rt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_rr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_rc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);

int cgemm_small_kernel_cn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_ct(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_cr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);
int cgemm_small_kernel_cc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb, float beta0, float beta1, float * C, BLASLONG ldc);

int zgemm_small_kernel_nn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_nt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_nr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_nc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
	
int zgemm_small_kernel_tn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_tt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_tr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_tc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);

int zgemm_small_kernel_rn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_rt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_rr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_rc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);

int zgemm_small_kernel_cn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_ct(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_cr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);
int zgemm_small_kernel_cc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb, double beta0, double beta1, double * C, BLASLONG ldc);

int cgemm_small_kernel_b0_nn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);	
int cgemm_small_kernel_b0_nt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_nr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_nc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
	
int cgemm_small_kernel_b0_tn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_tt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_tr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_tc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);

int cgemm_small_kernel_b0_rn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_rt(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_rr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_rc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);

int cgemm_small_kernel_b0_cn(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_ct(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_cr(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);
int cgemm_small_kernel_b0_cc(BLASLONG m, BLASLONG n, BLASLONG k, float * A, BLASLONG lda, float alpha0, float alpha1, float * B, BLASLONG ldb,  float * C, BLASLONG ldc);

int zgemm_small_kernel_b0_nn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);	
int zgemm_small_kernel_b0_nt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_nr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_nc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
	
int zgemm_small_kernel_b0_tn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_tt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_tr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_tc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);

int zgemm_small_kernel_b0_rn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_rt(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_rr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_rc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);

int zgemm_small_kernel_b0_cn(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_ct(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_cr(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);
int zgemm_small_kernel_b0_cc(BLASLONG m, BLASLONG n, BLASLONG k, double * A, BLASLONG lda, double alpha0, double alpha1, double * B, BLASLONG ldb,  double * C, BLASLONG ldc);


int sgemm_incopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_itcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_oncopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_otcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int dgemm_incopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int dgemm_itcopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int dgemm_oncopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int dgemm_otcopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int cgemm_incopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int cgemm_itcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int cgemm_oncopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int cgemm_otcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int zgemm_incopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int zgemm_itcopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int zgemm_oncopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);
int zgemm_otcopy(BLASLONG m, BLASLONG n, double *a, BLASLONG lda, double *b);

int sgemm_kernel(BLASLONG, BLASLONG, BLASLONG, float,  float  *, float  *, float  *, BLASLONG);
int dgemm_kernel(BLASLONG, BLASLONG, BLASLONG, double, double *, double *, double *, BLASLONG);

int cgemm_kernel_n(BLASLONG, BLASLONG, BLASLONG, float,  float,  float  *, float  *, float  *, BLASLONG);
int cgemm_kernel_l(BLASLONG, BLASLONG, BLASLONG, float,  float,  float  *, float  *, float  *, BLASLONG);
int cgemm_kernel_r(BLASLONG, BLASLONG, BLASLONG, float,  float,  float  *, float  *, float  *, BLASLONG);
int cgemm_kernel_b(BLASLONG, BLASLONG, BLASLONG, float,  float,  float  *, float  *, float  *, BLASLONG);

int zgemm_kernel_n(BLASLONG, BLASLONG, BLASLONG, double, double, double *, double *, double *, BLASLONG);
int zgemm_kernel_l(BLASLONG, BLASLONG, BLASLONG, double, double, double *, double *, double *, BLASLONG);
int zgemm_kernel_r(BLASLONG, BLASLONG, BLASLONG, double, double, double *, double *, double *, BLASLONG);
int zgemm_kernel_b(BLASLONG, BLASLONG, BLASLONG, double, double, double *, double *, double *, BLASLONG);

int sgemm_nn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int sgemm_nt(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int sgemm_tn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int sgemm_tt(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);

int dgemm_nn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int dgemm_nt(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int dgemm_tn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int dgemm_tt(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);

int cgemm_nn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_nt(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_nr(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_nc(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_tn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_tt(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_tr(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_tc(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_rn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_rt(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_rr(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_rc(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_cn(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_ct(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_cr(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);
int cgemm_cc(blas_arg_t *, BLASLONG *, BLASLONG *, float *, float *, BLASLONG);

int zgemm_nn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_nt(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_nr(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_nc(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_tn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_tt(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_tr(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_tc(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_rn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_rt(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_rr(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_rc(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_cn(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_ct(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_cr(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);
int zgemm_cc(blas_arg_t *, BLASLONG *, BLASLONG *, double *, double *, BLASLONG);

int sgemm_beta(BLASLONG, BLASLONG, BLASLONG, float,
	       float  *, BLASLONG, float   *, BLASLONG, float  *, BLASLONG);
int dgemm_beta(BLASLONG, BLASLONG, BLASLONG, double,
	       double *, BLASLONG, double  *, BLASLONG, double *, BLASLONG);
int cgemm_beta(BLASLONG, BLASLONG, BLASLONG, float,  float,
	       float  *, BLASLONG, float   *, BLASLONG, float  *, BLASLONG);
int zgemm_beta(BLASLONG, BLASLONG, BLASLONG, double, double,
	       double *, BLASLONG, double  *, BLASLONG, double *, BLASLONG);

#endif
