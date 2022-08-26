#ifdef GPU

#ifndef GPU_BLAS_H
#define GPU_BLAS_H

#include <vector>
#include <complex>

#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

inline void xaxpy_magma(const int N, const double alpha, 
		  const double* X, double* Y){
   int INCX = 1, INCY = 1;
   magma_daxpy(N, alpha, X, INCX, Y, INCY, magma_queue );
}
inline void xaxpy_magma(const int N, const std::complex<double> alpha, 
		  const std::complex<double>* X, std::complex<double>* Y){
   int INCX = 1, INCY = 1;
   magmaDoubleComplex alpha1{alpha.real(),alpha.imag()};
   magma_zaxpy(N, alpha1, (magmaDoubleComplex *)X, INCX, (magmaDoubleComplex *)Y, INCY, magma_queue);
}

} // linalg

#endif

#endif
