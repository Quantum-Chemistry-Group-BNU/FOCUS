#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"

template <typename Tm>
void test_xgemv_batch(const int M, const int N, const int nbatch){
   char* transa_array = new char[nbatch];
   transa_array[0] = 'N';
   MKL_INT* m_array = new MKL_INT[nbatch];
   MKL_INT* n_array = new MKL_INT[nbatch];
   Tm *alpha_array = new Tm[nbatch];
   Tm *beta_array = new Tm[nbatch];
   alpha_array[0] = 1.0;
   beta_array[0] = 0.0;
   MKL_INT *incx_array = new MKL_INT[nbatch];
   MKL_INT *incy_array = new MKL_INT[nbatch];
   incx_array[0] = 1;
   incy_array[0] = 1;
   MKL_INT group_count = nbatch;
   MKL_INT* group_size = new MKL_INT[nbatch];
   for(int i=0; i<nbatch; i++){
      m_array[i] = M;
      n_array[i] = N;
      group_size[i] = 1;
   }
   auto a = linalg::random_matrix<Tm>(M*N,nbatch);
   auto x = linalg::random_matrix<Tm>(N*1,nbatch);
   auto y = linalg::random_matrix<Tm>(M*1,nbatch);
   Tm** A_array = new Tm*[nbatch];
   Tm** X_array = new Tm*[nbatch];
   Tm** Y_array = new Tm*[nbatch];
   for(int i=0; i<nbatch; i++){
      A_array[i] = a.col(i);
      X_array[i] = x.col(i);
      Y_array[i] = y.col(i);
   }
   linalg::xgemv_batch(transa_array, m_array, n_array, alpha_array, A_array, m_array,
         X_array, incx_array, beta_array, Y_array, incy_array,
         &group_count, group_size);
}

int main(){
   
   int N0 = 100;
   int nbatch = 100;

   using Tm = double;

   std::cout << "GEMV_BATCH" << std::endl;
   for(int i=1; i<=10; i++){  
      int N = i*N0;
      auto t0 = tools::get_time();
      test_xgemv_batch<Tm>(N, N, nbatch);
      auto t1 = tools::get_time();
      double dt = tools::get_duration(t1-t0);
      std::cout << "i=" << i << " M,N=" << N << "," << N 
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
