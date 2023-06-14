#include <memory>
#include "core/matrix.h"
#include "core/blas.h"
#include "core/blas_batch.h"
#include "core/tools.h"
#ifndef SERIAL
#include "core/mpi_wrapper.h"
#endif
#ifdef GPU
#include "gpu/gpu_env.h"
#include "gpu/gpu_blas_batch.h"
#endif

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

template <typename Tm>
double test_xgemm_batch(const int M, const int N, const int K, const int nbatch){
   unique_ptr<char[]> transa_array(new char[nbatch]);
   unique_ptr<char[]> transb_array(new char[nbatch]);
   unique_ptr<MKL_INT[]> m_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> n_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> k_array(new MKL_INT[nbatch]);
   unique_ptr<Tm[]> alpha_array(new Tm[nbatch]);
   unique_ptr<Tm[]>  beta_array(new Tm[nbatch]);
   unique_ptr<MKL_INT[]> group_size(new MKL_INT[nbatch]);
   MKL_INT group_count = nbatch;
   for(int i=0; i<nbatch; i++){
      m_array[i] = M;
      n_array[i] = N;
      k_array[i] = K;
      transa_array[i] = 'N';
      transb_array[i] = 'N';
      alpha_array[i] = 1.0;
      beta_array[i] = 0.0;
      group_size[i] = 1;
   }
   auto a = linalg::random_matrix<Tm>(M*K,nbatch);
   auto b = linalg::random_matrix<Tm>(K*N,nbatch);
   auto c = linalg::random_matrix<Tm>(M*N,nbatch);
   std::vector<const Tm*> A_array(nbatch);
   std::vector<const Tm*> B_array(nbatch);
   std::vector<Tm*> C_array(nbatch);
   for(int i=0; i<nbatch; i++){
      A_array[i] = a.col(i);
      B_array[i] = b.col(i);
      C_array[i] = c.col(i);
   }
   auto t0 = tools::get_time();
   linalg::xgemm_batch(transa_array.get(), transb_array.get(),
         m_array.get(), n_array.get(), k_array.get(), 
         alpha_array.get(), A_array.data(), m_array.get(),
         B_array.data(), k_array.get(), beta_array.get(), 
         C_array.data(), m_array.get(),
         &group_count, group_size.get());
   auto t1 = tools::get_time();
   double dt = tools::get_duration(t1-t0);
   return dt;
}

#ifdef GPU

template <typename Tm>
double test_xgemv_batch_gpu(const int M, const int N, const int nbatch, const int iop){
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
   // gpu
   size_t total_dsize = nbatch*(M*N+N+M)*sizeof(Tm);
   void* dev_dtotal = GPUmem.allocate(total_dsize);
   Tm* dev_a = (Tm*)dev_dtotal;
   Tm* dev_x = dev_a + nbatch*M*N;
   Tm* dev_y = dev_x + nbatch*N*1; 
   GPUmem.to_gpu(dev_a, a.data(), nbatch*M*N*sizeof(Tm));
   GPUmem.to_gpu(dev_x, x.data(), nbatch*N*1*sizeof(Tm));
   GPUmem.to_gpu(dev_y, y.data(), nbatch*M*1*sizeof(Tm));
   std::vector<const Tm*> A_array(nbatch);
   std::vector<const Tm*> X_array(nbatch);
   std::vector<Tm*> Y_array(nbatch);
   for(int i=0; i<nbatch; i++){
      A_array[i] = dev_a + i*M*N;
      X_array[i] = dev_x + i*N;
      Y_array[i] = dev_y + i*M;
   }
   std::vector<int> gsta(2);
   gsta[0] = 0;
   gsta[1] = nbatch;
   auto t0 = tools::get_time();
   if(iop == 0){
      linalg::xgemv_batch_gpu_magma(transa_array[0], m_array.get(), n_array.get(), 
            alpha_array.get(), A_array.data(), m_array.get(),
            X_array.data(), incx_array.get(), beta_array.get(), 
            Y_array.data(), incy_array.get(),
            group_count);
   }else{
     linalg::xgemv_batch_gpu_grouped(transa_array[0], m_array.get(), n_array.get(), 
            alpha_array.get(), A_array.data(), m_array.get(),
            X_array.data(), incx_array.get(), beta_array.get(), 
            Y_array.data(), incy_array.get(),
            group_count, gsta);
   }
   auto t1 = tools::get_time();
   double dt = tools::get_duration(t1-t0);
   GPUmem.deallocate(dev_dtotal, total_dsize);
   return dt;
}

template <typename Tm>
double test_xgemm_batch_gpu(const int M, const int N, const int K, const int nbatch, const int iop){
   unique_ptr<char[]> transa_array(new char[nbatch]);
   unique_ptr<char[]> transb_array(new char[nbatch]);
   unique_ptr<MKL_INT[]> m_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> n_array(new MKL_INT[nbatch]);
   unique_ptr<MKL_INT[]> k_array(new MKL_INT[nbatch]);
   unique_ptr<Tm[]> alpha_array(new Tm[nbatch]);
   unique_ptr<Tm[]>  beta_array(new Tm[nbatch]);
   unique_ptr<MKL_INT[]> group_size(new MKL_INT[nbatch]);
   MKL_INT group_count = nbatch;
   for(int i=0; i<nbatch; i++){
      m_array[i] = M;
      n_array[i] = N;
      k_array[i] = K;
      transa_array[i] = 'N';
      transb_array[i] = 'N';
      alpha_array[i] = 1.0;
      beta_array[i] = 0.0;
      group_size[i] = 1;
   }
   auto a = linalg::random_matrix<Tm>(M*K,nbatch);
   auto b = linalg::random_matrix<Tm>(K*N,nbatch);
   auto c = linalg::random_matrix<Tm>(M*N,nbatch);
   size_t total_dsize = nbatch*(M*K+K*N+M*N)*sizeof(Tm);
   void* dev_dtotal = GPUmem.allocate(total_dsize);
   Tm* dev_a = (Tm*)dev_dtotal;
   Tm* dev_b = dev_a + nbatch*M*K;
   Tm* dev_c = dev_b + nbatch*K*N; 
   GPUmem.to_gpu(dev_a, a.data(), nbatch*M*K*sizeof(Tm));
   GPUmem.to_gpu(dev_b, b.data(), nbatch*K*N*sizeof(Tm));
   GPUmem.to_gpu(dev_c, c.data(), nbatch*M*N*sizeof(Tm));
   std::vector<const Tm*> A_array(nbatch);
   std::vector<const Tm*> B_array(nbatch);
   std::vector<Tm*> C_array(nbatch);
   for(int i=0; i<nbatch; i++){
      A_array[i] = dev_a + i*M*K;
      B_array[i] = dev_b + i*K*N;
      C_array[i] = dev_c + i*M*N;
   }
   std::vector<int> gsta(2);
   gsta[0] = 0;
   gsta[1] = nbatch;
   auto t0 = tools::get_time();
   if(iop == 0){
      linalg::xgemm_batch_gpu_magma(transa_array[0], transb_array[0],
            m_array.get(), n_array.get(), k_array.get(), 
            alpha_array.get(), A_array.data(), m_array.get(),
            B_array.data(), k_array.get(), beta_array.get(), 
            C_array.data(), m_array.get(),
            group_count);
   }else{
      linalg::xgemm_batch_gpu_grouped(transa_array[0], transb_array[0],
            m_array.get(), n_array.get(), k_array.get(), 
            alpha_array.get(), A_array.data(), m_array.get(),
            B_array.data(), k_array.get(), beta_array.get(), 
            C_array.data(), m_array.get(),
            group_count, gsta);
   }
   auto t1 = tools::get_time();
   double dt = tools::get_duration(t1-t0);
   GPUmem.deallocate(dev_dtotal, total_dsize);
   return dt;
}

#endif

int main(int argc, char *argv[]){
   
   int N0 = 300;
   int nbatch = 10;
   int imax = 10;

   using Tm = double;

   std::cout << "\n=== GEMV_BATCH ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemv_batch<Tm>(N, N, nbatch);
      std::cout << " i=" << i << " M,N=" << N << "," << N 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch<Tm>(N, N, 1, nbatch);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << 1 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch<Tm>(N, N, N, nbatch);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << N  
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N*N/dt
         << std::endl; 
   }

   int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
   // setup MPI environment 
   boost::mpi::environment env{argc, argv};
   boost::mpi::communicator world;
   rank = world.rank();
   size = world.size();
#endif

#ifdef GPU

   gpu_init(rank);

   std::cout << "\n=== GEMV_BATCH_GPU: magma ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemv_batch_gpu<Tm>(N, N, nbatch, 0);
      std::cout << " i=" << i << " M,N=" << N << "," << N 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMV_BATCH_GPU: cublas ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemv_batch_gpu<Tm>(N, N, nbatch, 1);
      std::cout << " i=" << i << " M,N=" << N << "," << N 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH: magma ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch_gpu<Tm>(N, N, 1, nbatch, 0);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << 1 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH: cublas ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch_gpu<Tm>(N, N, 1, nbatch, 1);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << 1 
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH: magma ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch_gpu<Tm>(N, N, N, nbatch, 0);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << N  
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N*N/dt
         << std::endl; 
   }

   std::cout << "\n=== GEMM_BATCH: cublas ===" << std::endl;
   for(int i=1; i<=imax; i+=2){  
      int N = i*N0;
      double dt = test_xgemm_batch_gpu<Tm>(N, N, N, nbatch, 1);
      std::cout << " i=" << i << " M,N,K=" << N << "," << N << "," << N  
         << " nbatch=" << nbatch
         << " time=" << dt
         << " flops=" << nbatch*2*double(N)*N*N/dt
         << std::endl; 
   }

   gpu_finalize();

#endif

  return 0;   
}
