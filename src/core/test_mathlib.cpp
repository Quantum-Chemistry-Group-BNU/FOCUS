#include <iostream>
#include "tools.h"
#include "matrix.h"
#include "ortho.h"
#include "linalg.h"
#include "tests_core.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ndim 50000000
#define rows 2000
#define cols 2000
#define cycle 3

using namespace std;
using namespace linalg;

template <typename Tm>
void test_loop(){
   std::vector<Tm> a(ndim,1);
   std::vector<Tm> b(ndim,1);
   double c = 1.2;
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
#ifdef _OPENMP
#pragma omp for schedule(static,1048576)
#endif 
      for(int i=0; i<ndim; i++){
         a[i] = b[i]/c;
      }
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_loop = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

template <typename Tm>
void test_xnrm2(){
   std::vector<Tm> a(ndim,1);
   double c = 1.2;
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      c = linalg::xnrm2(ndim, a.data());
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_xnrm2 = " << tools::get_duration(t1-t0)/cycle << " S c=" << c << std::endl;
}

template <typename Tm>
void test_xcopy(){
   std::vector<Tm> a(ndim,1);
   std::vector<Tm> b(ndim,1);
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      linalg::xcopy(ndim, a.data(), b.data());
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_xcopy = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

template <typename Tm>
void test_xscal(){
   std::vector<Tm> a(ndim,1);
   Tm c = 1.2;
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      linalg::xscal(ndim, c, a.data());
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_xscal = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

template <typename Tm>
void test_xgemm(){
   auto mat_A = linalg::random_matrix<Tm>(rows,cols);
   auto mat_B = linalg::random_matrix<Tm>(cols,rows);
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
     auto mat_C= xgemm("N","N",mat_A,mat_B);
   }
   auto t1 = tools::get_time();
   double tav = tools::get_duration(t1-t0)/cycle;
   std::cout << "time for test_xgemm = " << tav << " S"
    << " FLOPS=" << 2*double(rows)*double(cols)*double(rows)/tav/std::pow(1024.0,3) << " G/s"
    << std::endl;
}

template <typename Tm>
void test_ortho(){
   const int nres = 5;
   auto rbas = linalg::random_matrix<Tm>(ndim,nres);
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      int nindp = linalg::get_ortho_basis(ndim,nres,rbas.data());
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_ortho = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

template <typename Tm>
void test_eig(){
   auto mat = linalg::random_matrix<Tm>(rows,rows);
   mat = (mat + mat.H())*0.5;
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      vector<double> e(rows);
      matrix<double> v;
      eig_solver(mat, e, v);
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_eig = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

template <typename Tm>
void test_svd(){
   const int iop = 13;
   auto mat = linalg::random_matrix<Tm>(rows,rows);
   auto t0 = tools::get_time();
   for(int k=0; k<cycle; k++){
      vector<double> s;
      matrix<Tm> U, Vt;
      svd_solver(mat, s, U, Vt, iop);
   }
   auto t1 = tools::get_time();
   std::cout << "time for test_svd = " << tools::get_duration(t1-t0)/cycle << " S" << std::endl;
}

int tests::test_mathlib(){
   int maxthreads = 1;
#ifdef _OPENMP
   maxthreads = omp_get_max_threads();
#endif
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_mathlib maxthreads=" << maxthreads << endl;
   cout << "ndim=" << ndim << " rows=" << rows << " cols=" << cols << endl;
   cout << tools::line_separator << endl;

   using Tm = double;
   // loop
   test_loop<Tm>();
   // xnrm2
   test_xnrm2<Tm>();
   // xcopy
   test_xcopy<Tm>();
   // gemm
   test_xgemm<Tm>();
   // ortho
   test_ortho<Tm>();
   // eig
   test_eig<Tm>();
   // svd
   test_svd<Tm>();

/*
// mac machine - ZL@20230601
----------------------------------------------------------------------
tests::test_mathlib maxthreads=4
ndim=50000000 rows=2000 cols=2000
----------------------------------------------------------------------
time for test_loop = 0.203004 S
time for test_xnrm2 = 0.0417343 S c=7071.07
time for test_xcopy = 0.034611 S
time for test_xgemm = 0.0885083 S FLOPS=168.359 G/s
time for test_ortho = 2.64287 S
time for test_eig = 0.390302 S
time for test_svd = 1.69179 S
*/

   return 0;
}
