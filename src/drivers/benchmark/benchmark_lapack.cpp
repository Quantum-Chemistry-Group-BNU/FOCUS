#include <iostream>
#include "tools.h"
#include "matrix.h"
#include "ortho.h"
#include "linalg.h"
#include "tests_core.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef GPU
#include "gpu_linalg.h"
#endif

using namespace std;
using namespace linalg;

template <typename Tm>
void test_eig_cpu(const linalg::matrix<Tm>& mat){
   std::cout << "test_eig_cpu:";
   int rows = mat.rows();
   auto t0 = tools::get_time();
   {
      vector<double> e(rows);
      matrix<Tm> v;
      eig_solver(mat, e, v);
      std::cout << " e[0]=" << e[0] << " e[-1]=" << e[rows-1];
   }
   auto t1 = tools::get_time();
   std::cout << " time=" << tools::get_duration(t1-t0) << " S" << std::endl;
}

template <typename Tm>
void test_svd_cpu(const linalg::matrix<Tm>& mat,
      const int svd_iop=3){
   std::cout << "test_svd_cpu: svd_iop=" << svd_iop;
   int r = std::min(mat.rows(),mat.cols());
   auto t0 = tools::get_time();
   {
      vector<double> s;
      matrix<Tm> U, Vt;
      svd_solver(mat, s, U, Vt, svd_iop);
      std::cout << " s[0]=" << s[0] << " s[-1]=" << s[r-1];
   }
   auto t1 = tools::get_time();
   std::cout << " time=" << tools::get_duration(t1-t0) << " S" << std::endl;
}

#ifdef GPU
template <typename Tm>
void test_eig_gpu(const linalg::matrix<Tm>& mat){
   std::cout << "test_eig_gpu:";
   int rows = mat.rows();
   auto t0 = tools::get_time();
   {
      vector<double> e(rows);
      matrix<Tm> v;
      eig_solver_gpu(mat, e, v);
      std::cout << " e[0]=" << e[0] << " e[-1]=" << e[rows-1];
   }
   auto t1 = tools::get_time();
   std::cout << " time=" << tools::get_duration(t1-t0) << " S" << std::endl;
}

template <typename Tm>
void test_svd_gpu(const linalg::matrix<Tm>& mat,
      const int svd_iop=3){
   std::cout << "test_svd_gpu: svd_iop=" << svd_iop;
   int r = std::min(mat.rows(),mat.cols());
   auto t0 = tools::get_time();
   {
      vector<double> s;
      matrix<Tm> U, Vt;
      svd_solver_gpu(mat, s, U, Vt, svd_iop);
      std::cout << " s[0]=" << s[0] << " s[-1]=" << s[r-1];
   }
   auto t1 = tools::get_time();
   std::cout << " time=" << tools::get_duration(t1-t0) << " S" << std::endl;
}
#endif

int main(){
   int maxthreads = 1;
#ifdef _OPENMP
   maxthreads = omp_get_max_threads();
#endif
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "benchmark_lapack: eig & svd maxthreads=" << maxthreads << endl;
   cout << tools::line_separator << endl;

   const int cycle = 3;
   std::vector<int> rows({100,200,300,400,500,1000,1600,2000});
   std::vector<int> cols({100,250,350,400,300, 300, 100,2000});

   cout << "\n=== svd ===" << endl;
   for(int i=0; i<rows.size(); i++){
      std::cout << "\ni=" << i << " rows,cols=" << rows[i] << "," << cols[i] << std::endl;
      std::cout << "double:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = double;
         auto mat = linalg::random_matrix<Tm>(rows[i],cols[i]);
         test_svd_cpu<Tm>(mat, 3);
         test_svd_cpu<Tm>(mat, 13);
#ifdef GPU
         test_svd_gpu<Tm>(mat, 3);
         test_svd_gpu<Tm>(mat, 13);
#endif
      }
      std::cout << "complex:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = double;
         auto mat = linalg::random_matrix<Tm>(rows[i],cols[i]);
         test_svd_cpu<Tm>(mat, 3);
         test_svd_cpu<Tm>(mat, 13);
#ifdef GPU
         test_svd_gpu<Tm>(mat, 3);
         test_svd_gpu<Tm>(mat, 13);
#endif
      }
   }

   cout << "\n=== eig ===" << endl;
   for(int i=0; i<rows.size(); i++){
      std::cout << "\ni=" << i << " rows=" << rows[i] << std::endl;
      std::cout << "double:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = double;
         auto mat = linalg::random_matrix<Tm>(rows[i],rows[i]);
         mat = (mat + mat.H())*0.5;
         test_eig_cpu<Tm>(mat);
#ifdef GPU
         test_eig_gpu<Tm>(mat);
#endif
      }
      std::cout << "complex:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = std::complex<double>;
         auto mat = linalg::random_matrix<Tm>(rows[i],rows[i]);
         mat = (mat + mat.H())*0.5;
         test_eig_cpu<Tm>(mat);
#ifdef GPU
         test_eig_gpu<Tm>(mat);
#endif
      }
   }

#ifdef GPU
   // large tests
   rows = {1000,2000,3000,4000,5000,6000,10000};

   cout << "\n=== svd[large] ===" << endl;
   for(int i=0; i<rows.size(); i++){
      int cols = rows[i]/1.8;
      std::cout << "\ni=" << i << " rows,cols=" << rows[i] << "," << cols << std::endl;
      std::cout << "double:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = double;
         auto mat = linalg::random_matrix<Tm>(rows[i],cols);
         test_svd_cpu<Tm>(mat, 3);
         test_svd_cpu<Tm>(mat, 13);
         test_svd_gpu<Tm>(mat, 3);
         test_svd_gpu<Tm>(mat, 13);
      }
   }

   cout << "\n=== eig[large] ===" << endl;
   for(int i=0; i<rows.size(); i++){
      std::cout << "\ni=" << i << " rows=" << rows[i] << std::endl;
      std::cout << "double:" << std::endl;
      for(int k=0; k<cycle; k++){
         using Tm = double;
         auto mat = linalg::random_matrix<Tm>(rows[i],rows[i]);
         mat = (mat + mat.H())*0.5;
         test_eig_gpu<Tm>(mat);
      }
   }
#endif

   return 0;
}
