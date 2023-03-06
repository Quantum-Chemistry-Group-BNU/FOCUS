#ifndef BLAS_BATCH_H
#define BLAS_BATCH_H

extern "C" {

void dgemm_batch_(const char *transa_array, const char *transb_array, 
        const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
        const double *alpha_array, const double **a_array, const MKL_INT *lda_array, 
        const double **b_array, const MKL_INT *ldb_array,
        const double *beta_array, double **c_array, const MKL_INT *ldc_array, 
        const MKL_INT *group_count, const MKL_INT *group_size);

void zgemm_batch_(const char *transa_array, const char *transb_array, 
        const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
        const std::complex<double> *alpha_array, const std::complex<double> **a_array, 
        const MKL_INT *lda_array,
        const std::complex<double> **b_array, const MKL_INT *ldb_array,
        const std::complex<double> *beta_array, std::complex<double> **c_array, 
        const MKL_INT *ldc_array, 
        const MKL_INT *group_count, const MKL_INT *group_size);

}

// wrapper for BLAS
namespace linalg{

inline void xgemm_batch(const char *transa_array, const char *transb_array, 
        const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
        const double *alpha_array, const double **a_array, const MKL_INT *lda_array, 
        const double **b_array, const MKL_INT *ldb_array,
        const double *beta_array, double **c_array, const MKL_INT *ldc_array, 
        const MKL_INT *group_count, const MKL_INT *group_size)
{
    return ::dgemm_batch_(
            transa_array, transb_array, 
            m_array, n_array, k_array,
            alpha_array, a_array, lda_array, 
            b_array, ldb_array,
            beta_array, c_array, ldc_array, 
            group_count, group_size
            );
}

inline void xgemm_batch(const char *transa_array, const char *transb_array, 
        const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
        const std::complex<double> *alpha_array, 
        const std::complex<double> **a_array, const MKL_INT *lda_array,
        const std::complex<double> **b_array, const MKL_INT *ldb_array,
        const std::complex<double> *beta_array, std::complex<double> **c_array, 
        const MKL_INT *ldc_array, 
        const MKL_INT *group_count, const MKL_INT *group_size)
{
    return ::zgemm_batch_(
            transa_array, transb_array, 
            m_array, n_array, k_array,
            alpha_array, a_array, lda_array, 
            b_array, ldb_array,
            beta_array, c_array, ldc_array, 
            group_count, group_size
            );
}

} // linalg

#endif
