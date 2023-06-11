#include <memory>
#include "../core/matrix.h"
#include "../core/blas.h"
#include "../core/blas_batch.h"
#include "../core/tools.h"

using namespace std;

template <typename Tm>
double test_xgemv_batch(const int M, const int N, const int nbatch){
   unique_ptr<char[]> transa_array(new char[nbatch]);
   unique_ptr<MKL_INT[]> m_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> n_array(new MKL_INT[nbatch]);
   unique_ptr<Tm[]> alpha_array(new Tm[nbatch]);
   unique_ptr<Tm[]>  beta_array(new Tm[nbatch]);
   unique_ptr<MKL_INT[]> incx_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> incy_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> group_size(new MKL_INT[nbatch]);
   MKL_INT group_count = nbatch;
   for(int i=0; i<nbatch; i++){
      m_array[i] = M;
      n_array[i] = N;
      transa_array[i] = 'N';
      alpha_array[i] = 1.0;
      beta_array[i] = 0.0;
      incx_array[i] = 1;
      incy_array[i] = 1;
      group_size[i] = 1;
   }
   auto a = linalg::random_matrix<Tm>(M*N,nbatch);
   auto x = linalg::random_matrix<Tm>(N*1,nbatch);
   auto y = linalg::random_matrix<Tm>(M*1,nbatch);
   std::vector<const Tm*> A_array(nbatch);
   std::vector<const Tm*> X_array(nbatch);
   std::vector<Tm*> Y_array(nbatch);
   for(int i=0; i<nbatch; i++){
      A_array[i] = a.col(i);
      X_array[i] = x.col(i);
      Y_array[i] = y.col(i);
   }
   auto t0 = tools::get_time();
   linalg::xgemv_batch(transa_array.get(), m_array.get(), n_array.get(), 
         alpha_array.get(), A_array.data(), m_array.get(),
         X_array.data(), incx_array.get(), beta_array.get(), 
         Y_array.data(), incy_array.get(),
         &group_count, group_size.get());
   auto t1 = tools::get_time();
   double dt = tools::get_duration(t1-t0);
   return dt;
}

int main(){
   
   int N0 = 100;
   int nbatch = 100;

   using Tm = double;

   std::cout << "\nGEMV_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      double dt = test_xgemv_batch<Tm>(N, N, nbatch);
      std::cout << " i=" << i << " M,N=" << N << "," << N 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }
/*
   std::cout << "GEMM_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemm_batch<Tm>(N, N, 1, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N,K=" << N << "," << N << "," << 1 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }
   std::cout << "GEMM_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemm_batch<Tm>(N, N, N, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N,K=" << N << "," << N << "," << N  
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N*N/dt
         << std::endl; 
   }

#ifdef GPU
   std::cout << "GEMV_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemv_batch_gpu<Tm>(N, N, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N=" << N << "," << N 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }
   std::cout << "GEMM_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemm_batch_gpu<Tm>(N, N, 1, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N,K=" << N << "," << N << "," << 1 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }
   std::cout << "GEMM_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemm_batch_gpu<Tm>(N, N, N, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N,K=" << N << "," << N << "," << N  
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N*N/dt
         << std::endl; 
   }
#endif
*/

  return 0;   
}
