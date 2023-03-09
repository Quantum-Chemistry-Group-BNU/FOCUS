//
// ZQUATEV: Diagonalization of quaternionic matrices
// File   : f77.h
// Copyright (c) 2013, Toru Shiozaki (shiozaki@northwestern.edu)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// The views and conclusions contained in the software and documentation are those
// of the authors and should not be MKL_INTerpreted as representing official policies,
// either expressed or implied, of the FreeBSD Project.
//

#ifndef __TS_ZQUATEV_F77_H
#define __TS_ZQUATEV_F77_H

#include <complex>

#ifdef MKL_ILP64
   #define MKL_INT long long int
#endif

// blas
extern "C" {

 void zscal_(const MKL_INT*, const std::complex<double>*, std::complex<double>*, const MKL_INT*);
#ifndef ZDOT_RETURN
 void zdotc_(std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*);
#else
 std::complex<double> zdotc_(const MKL_INT*, const std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*);
#endif
 void zaxpy_(const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, std::complex<double>*, const MKL_INT*);
 void zgemv_(const char*, const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*,
             const std::complex<double>*, std::complex<double>*, const MKL_INT*);
 void ztrmv_(const char*, const char*, const char*, const MKL_INT*, const std::complex<double>*, const MKL_INT*, std::complex<double>*, const MKL_INT*);
#ifdef HAVE_ZGEMM3M
 void zgemm3m_(const char* transa, const char* transb, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
               const std::complex<double>* alpha, const std::complex<double>* a, const MKL_INT* lda, const std::complex<double>* b, const MKL_INT* ldb,
               const std::complex<double>* beta, std::complex<double>* c, const MKL_INT* ldc);
#else
 void zgemm_(const char* transa, const char* transb, const MKL_INT* m, const MKL_INT* n, const MKL_INT* k,
             const std::complex<double>* alpha, const std::complex<double>* a, const MKL_INT* lda, const std::complex<double>* b, const MKL_INT* ldb,
             const std::complex<double>* beta, std::complex<double>* c, const MKL_INT* ldc);
#endif
 void zrot_(const MKL_INT*, std::complex<double>*, const MKL_INT*, std::complex<double>*, const MKL_INT*, const double*, const std::complex<double>*);
 void zgerc_(const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*,
             std::complex<double>*, const MKL_INT*);
 void zgeru_(const MKL_INT*, const MKL_INT*, const std::complex<double>*, const std::complex<double>*, const MKL_INT*, const std::complex<double>*, const MKL_INT*,
             std::complex<double>*, const MKL_INT*);

 // lapack
 void zheev_(const char*, const char*, const MKL_INT*, std::complex<double>*, const MKL_INT*, double*, std::complex<double>*, const MKL_INT*, double*, MKL_INT*);
 void zhbev_(const char*, const char*, const MKL_INT*, const MKL_INT*, std::complex<double>*, const MKL_INT*, double*, std::complex<double>*, const MKL_INT*,
             std::complex<double>*, double*, MKL_INT*);
 void zlartg_(const std::complex<double>*, const std::complex<double>*, double*, std::complex<double>*, std::complex<double>*);
 void zlarfg_(const MKL_INT*, const std::complex<double>*, std::complex<double>*, const MKL_INT*, std::complex<double>*);
}


namespace {

 void zgemv_(const char* a, const MKL_INT b, const MKL_INT c, const std::complex<double> d, const std::complex<double>* e, const MKL_INT f, const std::complex<double>* g, const MKL_INT h,
             const std::complex<double> i, std::complex<double>* j, const MKL_INT k) { ::zgemv_(a,&b,&c,&d,e,&f,g,&h,&i,j,&k); }
 void ztrmv_(const char* a, const char* b, const char* c, const MKL_INT d, const std::complex<double>* e, const MKL_INT f, std::complex<double>* g, const MKL_INT h)
            { ::ztrmv_(a,b,c,&d,e,&f,g,&h); }
#ifdef HAVE_ZGEMM3M
 void zgemm3m_(const char* transa, const char* transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
               const std::complex<double> alpha, const std::complex<double>* a, const MKL_INT lda, const std::complex<double>* b, const MKL_INT ldb,
               const std::complex<double> beta, std::complex<double>* c, const MKL_INT ldc) { ::zgemm3m_(transa,transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc); }
#else
 void zgemm3m_(const char* transa, const char* transb, const MKL_INT m, const MKL_INT n, const MKL_INT k,
             const std::complex<double> alpha, const std::complex<double>* a, const MKL_INT lda, const std::complex<double>* b, const MKL_INT ldb,
             const std::complex<double> beta, std::complex<double>* c, const MKL_INT ldc) { ::zgemm_(transa,transb,&m,&n,&k,&alpha,a,&lda,b,&ldb,&beta,c,&ldc); }
#endif

 void zaxpy_(const MKL_INT a, const std::complex<double> b, const std::complex<double>* c, const MKL_INT d, std::complex<double>* e, const MKL_INT f) { ::zaxpy_(&a,&b,c,&d,e,&f); }
 void zscal_(const MKL_INT a, const std::complex<double> b, std::complex<double>* c, const MKL_INT d) { ::zscal_(&a, &b, c, &d); }
#ifndef ZDOT_RETURN
 std::complex<double> zdotc_(const MKL_INT b, const std::complex<double>* c, const MKL_INT d, const std::complex<double>* e, const MKL_INT f) {
   std::complex<double> a;
   ::zdotc_(&a,&b,c,&d,e,&f);
   return a;
 }
#else
 std::complex<double> zdotc_(const MKL_INT a, const std::complex<double>* b, const MKL_INT c, const std::complex<double>* d, const MKL_INT e) { return ::zdotc_(&a,b,&c,d,&e); }
#endif
 void zheev_(const char* a, const char* b, const MKL_INT c, std::complex<double>* d, const MKL_INT e, double* f, std::complex<double>* g, const MKL_INT h, double* i, MKL_INT& j)
             { ::zheev_(a,b,&c,d,&e,f,g,&h,i,&j); }
 void zhbev_(const char* a, const char* b, const MKL_INT c, const MKL_INT d, std::complex<double>* e, const MKL_INT f, double* g, std::complex<double>* h, const MKL_INT i,
             std::complex<double>* j, double* k, MKL_INT& l) { ::zhbev_(a, b, &c, &d, e, &f, g, h, &i, j, k, &l); }
 void zrot_(const MKL_INT a, std::complex<double>* b, const MKL_INT c, std::complex<double>* d, const MKL_INT e, const double f, const std::complex<double> g) {
            ::zrot_(&a, b, &c, d, &e, &f, &g); }
 void zgerc_(const MKL_INT a, const MKL_INT b, const std::complex<double> c, const std::complex<double>* d, const MKL_INT e, const std::complex<double>* f, const MKL_INT g,
             std::complex<double>* h, const MKL_INT i) { ::zgerc_(&a, &b, &c, d, &e, f, &g, h, &i); }
 void zgeru_(const MKL_INT a, const MKL_INT b, const std::complex<double> c, const std::complex<double>* d, const MKL_INT e, const std::complex<double>* f, const MKL_INT g,
             std::complex<double>* h, const MKL_INT i) { ::zgeru_(&a, &b, &c, d, &e, f, &g, h, &i); }
 void zlartg_(const std::complex<double> a, const std::complex<double> b, double& c, std::complex<double>& d, std::complex<double>& e) { ::zlartg_(&a, &b, &c, &d, &e); }
 void zlarfg_(const MKL_INT a, std::complex<double>& b, std::complex<double>* c, const MKL_INT d, std::complex<double>& e) { ::zlarfg_(&a, &b, c, &d, &e); }

}

#endif
