#ifdef GPU

#ifndef GPU_BLAS_BATCH_H
#define GPU_BLAS_BATCH_H

#include "gpu_env.h"

// wrapper for BLAS_BATCH
namespace linalg{

   // --- GEMV ---
#ifdef MAGMA
   // double
   inline void xgemv_batch_gpu_magma(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const double *alpha, const double **A_array, const MKL_INT *lda_array, 
         const double **X_array, const MKL_INT *incx_array,
         const double *beta, double** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_magma"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_lda,dev_incx,dev_incy
      size_t total_isize = 5*(batch_count+1)*sizeof(MKL_INT);
      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      MKL_INT* dev_m = (MKL_INT*)dev_itotal;
      MKL_INT* dev_n = dev_m + (batch_count+1);
      MKL_INT* dev_lda = dev_n + (batch_count+1);
      MKL_INT* dev_incx = dev_lda + (batch_count+1);
      MKL_INT* dev_incy = dev_incx + (batch_count+1);
      double** dev_A_array= (double**)dev_dtotal;
      double** dev_X_array = dev_A_array + batch_count;
      double** dev_Y_array = dev_X_array + batch_count;

      GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_incx, incx_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_incy, incy_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_A_array, A_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_X_array, X_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_Y_array, Y_array, batch_count*sizeof(double*));

      magma_trans_t Trans = MagmaNoTrans;
      if(trans=='T'){
         Trans = MagmaTrans;
      }else if (trans == 'C'){
         Trans = MagmaConjTrans;
      }

      magmablas_dgemv_vbatched(
               Trans,
               dev_m,
               dev_n,
               alpha[0],
               dev_A_array,
               dev_lda,
               dev_X_array,
               dev_incx,
               beta[0],
               dev_Y_array,
               dev_incy,
               batch_count,
               magma_queue
               );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }

   // complex 
   inline void xgemv_batch_gpu_magma(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **A_array, const MKL_INT *lda_array, 
         const std::complex<double> **X_array, const MKL_INT *incx_array,
         const std::complex<double> *beta, std::complex<double>** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_magma"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_lda,dev_incx,dev_incy
      size_t total_isize = 5*(batch_count+1)*sizeof(MKL_INT);
      size_t total_dsize = 3*batch_count*sizeof(magmaDoubleComplex*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      MKL_INT* dev_m = (MKL_INT*)dev_itotal;
      MKL_INT* dev_n = dev_m + (batch_count+1);
      MKL_INT* dev_lda = dev_n + (batch_count+1);
      MKL_INT* dev_incx = dev_lda + (batch_count+1);
      MKL_INT* dev_incy = dev_incx + (batch_count+1);
      magmaDoubleComplex** dev_A_array= (magmaDoubleComplex**)dev_dtotal;
      magmaDoubleComplex** dev_X_array = dev_A_array + batch_count;
      magmaDoubleComplex** dev_Y_array = dev_X_array + batch_count;

      GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_incx, incx_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_incy, incy_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_A_array, A_array, batch_count*sizeof(magmaDoubleComplex*));
      GPUmem.to_gpu(dev_X_array, X_array, batch_count*sizeof(magmaDoubleComplex*));
      GPUmem.to_gpu(dev_Y_array, Y_array, batch_count*sizeof(magmaDoubleComplex*));

      magma_trans_t Trans = MagmaNoTrans;
      if(trans=='T'){
         Trans = MagmaTrans;
      }else if (trans == 'C'){
         Trans = MagmaConjTrans;
      }
      magmaDoubleComplex alpha1{alpha->real(),alpha->imag()};
      magmaDoubleComplex beta1{beta->real(),beta->imag()};

      magmablas_zgemv_vbatched(
               Trans,
               dev_m,
               dev_n,
               alpha1,
               dev_A_array,
               dev_lda,
               dev_X_array,
               dev_incx,
               beta1,
               dev_Y_array,
               dev_incy,
               batch_count,
               magma_queue
               );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }
#endif

#ifndef USE_HIP
   // --- GEMV GROUPED ---
   // double
   inline void xgemv_batch_gpu_grouped(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const double *alpha, const double **A_array, const MKL_INT *lda_array, 
         const double **X_array, const MKL_INT *incx_array,
         const double *beta, double** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_group"<<std::endl;
         return ;
      }

      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_dtotal = GPUmem.allocate(total_dsize);
      double** dev_A_array= (double**)dev_dtotal;
      double** dev_X_array = dev_A_array + batch_count;
      double** dev_Y_array = dev_X_array + batch_count;
      GPUmem.to_gpu(dev_A_array, A_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_X_array, X_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_Y_array, Y_array, batch_count*sizeof(double*));

      for(int i=0; i<gsta.size()-1; i++){
         int ista = gsta[i];
         int nbatch = gsta[i+1]-ista;
         // convert from MKL_INT to int 
         int m = m_array[ista], n = n_array[ista];
         int lda = lda_array[ista], incx = incx_array[ista], incy = incy_array[ista]; 
         cublasOperation_t Trans = CUBLAS_OP_N ;
         if(trans=='T' || trans=='C'){
            Trans = CUBLAS_OP_T;
         }
//         // https://docs.nvidia.com/cuda/cublas/index.html
//         CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
//                  Trans, 
//                  m, n,
//                  alpha,
//                  &dev_A_array[ista], lda, // pointer of matrix should be on device
//                  &dev_X_array[ista], incx,
//                  beta,
//                  &dev_Y_array[ista], incy,
//                  nbatch));

            //xiangchunyang 20250103
#if 0
         // https://docs.nvidia.com/cuda/cublas/index.html
         CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
                  Trans, 
                  m, n,
                  alpha,
                  &dev_A_array[ista], lda, // pointer of matrix should be on device
                  &dev_X_array[ista], incx,
                  beta,
                  &dev_Y_array[ista], incy,
                  nbatch));

#else
           
           //for(int j=0; j<nbatch; j++){
           //   cublasDgemv(handle_cublas,
           //         Trans, 
           //         m, n, 
           //         alpha,
           //         A_array[ista+j], lda,
           //         X_array[ista+j], incx,
           //         beta,
           //         Y_array[ista+j], incy);
           //}
           magma_trans_t Trans_ = MagmaNoTrans;
           if(trans=='T' || trans=='C'){
               Trans = CUBLAS_OP_T;
               Trans_ = MagmaTrans;
           }

           magmablas_dgemv_batched(
                              Trans_, 
                              m, n,
                              alpha[0],
                              &dev_A_array[ista], lda, // pointer should be on device
                              &dev_X_array[ista], incx,
                              beta[0],
                              &dev_Y_array[ista], incy,
                              nbatch,
                              magma_queue
                              );
#endif
         /*
            for(int j=0; j<nbatch; j++){
            std::cout << "ista=" << ista << " j=" << j << " ista+j=" << ista+j 
            << " m,n,lda=" << m << "," << n << "," << lda 
            << " m,n,lda=" << m_array[ista+j] << "," << n_array[ista+j] << "," << lda_array[ista+j] 
            << std::endl;
            if(m != m_array[ista+j]){ std::cout << "error1" << std::endl; }
            if(n != n_array[ista+j]){ std::cout << "error2" << std::endl; }
            if(lda != lda_array[ista+j]){ std::cout << "error3" << std::endl; }
            cublasDgemv(handle_cublas,
            Trans, 
            m, n, 
            alpha,
            A_array[ista+j], lda,
            X_array[ista+j], incx,
            beta,
            Y_array[ista+j], incy);
            }
            */
      } // group

      GPUmem.deallocate(dev_dtotal, total_dsize);
   }

   // complex 
   inline void xgemv_batch_gpu_grouped(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **A_array, const MKL_INT *lda_array, 
         const std::complex<double> **X_array, const MKL_INT *incx_array,
         const std::complex<double> *beta, std::complex<double>** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_grouped"<<std::endl;
         return ;
      }
      std::cout << "COMPLEX CASE IS NOT IMPLEMENTED YET!" << std::endl;
      exit(1);
   }

   // --- GEMV STREAM ---
   // double
   inline void xgemv_batch_gpu_stream(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const double *alpha, const double **A_array, const MKL_INT *lda_array, 
         const double **X_array, const MKL_INT *incx_array,
         const double *beta, double** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_stream"<<std::endl;
         return ;
      }

      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_dtotal = GPUmem.allocate(total_dsize);
      double** dev_A_array= (double**)dev_dtotal;
      double** dev_X_array = dev_A_array + batch_count;
      double** dev_Y_array = dev_X_array + batch_count;
      GPUmem.to_gpu(dev_A_array, A_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_X_array, X_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_Y_array, Y_array, batch_count*sizeof(double*));

      size_t gsize = gsta.size()-1;
      int ntimes = (gsize+NSTREAMS-1)/NSTREAMS; 
      for(int k=0; k<ntimes; k++){
         size_t off = k*NSTREAMS;
         size_t jlen = std::min(gsize-off, size_t(NSTREAMS));

         for(int j=0; j<jlen; j++){
            size_t jdx = off+j;
            CUBLAS_CHECK(cublasSetStream(handle_cublas, custream[j])); 

            int ista = gsta[jdx];
            int nbatch = gsta[jdx+1]-ista;
            // convert from MKL_INT to int 
            int m = m_array[ista], n = n_array[ista];
            int lda = lda_array[ista], incx = incx_array[ista], incy = incy_array[ista]; 
            cublasOperation_t Trans = CUBLAS_OP_N ;
            if(trans=='T' || trans=='C'){
               Trans = CUBLAS_OP_T;
            }
//            // https://docs.nvidia.com/cuda/cublas/index.html
//            CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
//                     Trans, 
//                     m, n,
//                     alpha,
//                     &dev_A_array[ista], lda, // pointer of matrix should be on device
//                     &dev_X_array[ista], incx,
//                     beta,
//                     &dev_Y_array[ista], incy,
//                     nbatch));
//
            //xiangchunyang 20250103
#if 0
            // https://docs.nvidia.com/cuda/cublas/index.html
            CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
                     Trans, 
                     m, n,
                     alpha,
                     &dev_A_array[ista], lda, // pointer of matrix should be on device
                     &dev_X_array[ista], incx,
                     beta,
                     &dev_Y_array[ista], incy,
                     nbatch));
#else
          // for(int jj=0; jj<nbatch; jj++){
          //    cublasDgemv(handle_cublas,
          //          Trans, 
          //          m, n, 
          //          alpha,
          //          A_array[ista+jj], lda,
          //          X_array[ista+jj], incx,
          //          beta,
          //          Y_array[ista+jj], incy);
          // }
           magma_trans_t Trans_ = MagmaNoTrans;
           if(trans=='T' || trans=='C'){
               Trans = CUBLAS_OP_T;
               Trans_ = MagmaTrans;
           }
              magmablas_dgemv_batched(
                      Trans_, 
                      m, n,
                      alpha[0],
                      &dev_A_array[ista], lda, // pointer should be on device
                      &dev_X_array[ista], incx,
                      beta[0],
                      &dev_Y_array[ista], incy,
                      nbatch,
                      magma_queue
                      );
#endif
         } // j

         for(int j=0; j<jlen; j++){
            CUDA_CHECK(cudaStreamSynchronize(custream[j]));
         }
      } // k

      GPUmem.deallocate(dev_dtotal, total_dsize);
   }

   // complex 
   inline void xgemv_batch_gpu_stream(const char trans,
         const MKL_INT *m_array, const MKL_INT *n_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **A_array, const MKL_INT *lda_array, 
         const std::complex<double> **X_array, const MKL_INT *incx_array,
         const std::complex<double> *beta, std::complex<double>** Y_array, const MKL_INT *incy_array,
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_stream"<<std::endl;
         return ;
      }
      std::cout << "COMPLEX CASE IS NOT IMPLEMENTED YET!" << std::endl;
      exit(1);
   }
#endif

   // --- GEMM ---
#ifdef MAGMA
   // double
   inline void xgemm_batch_gpu_magma(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const double *alpha, const double **a_array, const MKL_INT *lda_array, 
         const double **b_array, const MKL_INT *ldb_array,
         const double *beta, double **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_magma"<<std::endl;
         return ;
      }

      //auto t0 = tools::get_time();

      //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
      size_t total_isize = 6*(batch_count+1)*sizeof(MKL_INT);
      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      //auto t1 = tools::get_time();

      MKL_INT* dev_m = (MKL_INT*)dev_itotal;
      MKL_INT* dev_n = dev_m + (batch_count+1);
      MKL_INT* dev_k = dev_n + (batch_count+1);
      MKL_INT* dev_lda = dev_k + (batch_count+1);
      MKL_INT* dev_ldb = dev_lda + (batch_count+1);
      MKL_INT* dev_ldc = dev_ldb + (batch_count+1);
      double** dev_a_array = (double**)dev_dtotal;
      double** dev_b_array = dev_a_array + batch_count;
      double** dev_c_array = dev_b_array + batch_count;

      GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_k, k_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_ldb, ldb_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_ldc, ldc_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));

      //auto t2 = tools::get_time();

      magma_trans_t transA = MagmaNoTrans;
      magma_trans_t transB = MagmaNoTrans;
      if(transa=='T'){
         transA = MagmaTrans;
      }else if(transa == 'C'){
         transA = MagmaConjTrans;
      }
      if(transb=='T'){
         transB = MagmaTrans;
      }else if(transb == 'C'){
         transB = MagmaConjTrans;
      }

      magmablas_dgemm_vbatched(
               transA,
               transB,
               dev_m,
               dev_n,
               dev_k,
               alpha[0],
               dev_a_array,
               dev_lda,
               dev_b_array,
               dev_ldb,
               beta[0],
               dev_c_array,
               dev_ldc,
               batch_count,
               magma_queue
               );

      //auto t3 = tools::get_time();

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);

      /*
         auto t4 = tools::get_time();
         std::cout << "GEMM[magma]: talloc=" << tools::get_duration(t1-t0)
         << " t2gpu=" << tools::get_duration(t2-t1)
         << " tcomp=" << tools::get_duration(t3-t2)
         << " tdealloc=" << tools::get_duration(t4-t3)
         << " total=" << tools::get_duration(t4-t0)
         << std::endl;
         */
   }

   // complex
   inline void xgemm_batch_gpu_magma(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **a_array, const MKL_INT *lda_array,
         const std::complex<double> **b_array, const MKL_INT *ldb_array,
         const std::complex<double> *beta, std::complex<double> **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_magma"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
      size_t total_isize = 6*(batch_count+1)*sizeof(MKL_INT);
      size_t total_dsize = 3*batch_count*sizeof(magmaDoubleComplex*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      MKL_INT* dev_m = (MKL_INT*)dev_itotal;
      MKL_INT* dev_n = dev_m + (batch_count+1);
      MKL_INT* dev_k = dev_n + (batch_count+1);
      MKL_INT* dev_lda = dev_k + (batch_count+1);
      MKL_INT* dev_ldb = dev_lda + (batch_count+1);
      MKL_INT* dev_ldc = dev_ldb + (batch_count+1);
      magmaDoubleComplex** dev_a_array = (magmaDoubleComplex**)dev_dtotal;
      magmaDoubleComplex** dev_b_array = dev_a_array + batch_count;
      magmaDoubleComplex** dev_c_array = dev_b_array + batch_count;

      GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_k, k_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_ldb, ldb_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_ldc, ldc_array, batch_count*sizeof(MKL_INT));
      GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(magmaDoubleComplex*));
      GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(magmaDoubleComplex*));
      GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(magmaDoubleComplex*));

      magma_trans_t transA = MagmaNoTrans;
      magma_trans_t transB = MagmaNoTrans;
      if(transa=='T'){
         transA = MagmaTrans;
      }else if (transa == 'C'){
         transA = MagmaConjTrans;
      }
      if(transb=='T'){
         transB = MagmaTrans;
      }else if(transb == 'C'){
         transB = MagmaConjTrans;
      }
      magmaDoubleComplex alpha1{alpha->real(),alpha->imag()};
      magmaDoubleComplex beta1{beta->real(),beta->imag()};

      magmablas_zgemm_vbatched(
               transA,
               transB,
               dev_m,
               dev_n,
               dev_k,
               alpha1,
               dev_a_array,
               dev_lda,
               dev_b_array,
               dev_ldb,
               beta1,
               dev_c_array,
               dev_ldc,
               batch_count,
               magma_queue
               );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }
#endif // MAGMA

#ifdef GEMMGROUPED

   // for CUBLAS version >= 12.4
   // https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Level-3/gemmGroupedBatched
   // --- GEMM GROUPED ---
   inline void xgemm_batch_gpu_grouped(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const double *alpha, const double **a_array, const MKL_INT *lda_array, 
         const double **b_array, const MKL_INT *ldb_array,
         const double *beta, double **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_grouped"<<std::endl;
         return ;
      }

      auto t0 = tools::get_time();

      using cublas_int = int;
      cublas_int group_count = gsta.size()-1;
      std::vector<cublas_int> group_size(group_count);
      std::vector<cublas_int> im_array(group_count);
      std::vector<cublas_int> in_array(group_count);
      std::vector<cublas_int> ik_array(group_count);
      std::vector<cublas_int> ilda_array(group_count);
      std::vector<cublas_int> ildb_array(group_count);
      std::vector<cublas_int> ildc_array(group_count);
      std::vector<double> alpha_array(group_count);
      std::vector<double> beta_array(group_count);	
      std::vector<cublasOperation_t> transa_array(group_count);
      std::vector<cublasOperation_t> transb_array(group_count);
      double cost = 0.0;
      for(int i=0; i<group_count; i++){
         group_size[i] = gsta[i+1]-gsta[i];
         // convert from MKL_INT to int 
         int ista = gsta[i];
         im_array[i] = m_array[ista];
         in_array[i] = n_array[ista];
         ik_array[i] = k_array[ista];
         ilda_array[i] = lda_array[ista];
         ildb_array[i] = ldb_array[ista];
         ildc_array[i] = ldc_array[ista]; 
         alpha_array[i] = alpha[ista];
         beta_array[i] = beta[ista];
         transa_array[i] = (transa=='T' || transa=='C')? CUBLAS_OP_T : CUBLAS_OP_N;
         transb_array[i] = (transb=='T' || transb=='C')? CUBLAS_OP_T : CUBLAS_OP_N;
         cost += 2.0*double(m_array[i])*n_array[i]*k_array[i]*group_size[i];
      }

      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_dtotal = GPUmem.allocate(total_dsize); // a,b,c
      auto t1 = tools::get_time();

      double** dev_a_array = (double**)dev_dtotal;
      double** dev_b_array = dev_a_array + batch_count;
      double** dev_c_array = dev_b_array + batch_count;
      GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));
      auto t2 = tools::get_time();

      CUBLAS_CHECK(cublasDgemmGroupedBatched(handle_cublas,
               transa_array.data(), transb_array.data(),
               im_array.data(), in_array.data(), ik_array.data(),
               alpha_array.data(),
               dev_a_array, ilda_array.data(), // pointer of matrix should be on device
               dev_b_array, ildb_array.data(),
               beta_array.data(),
               dev_c_array, ildc_array.data(),
               group_count,
               group_size.data()));
      auto t3 = tools::get_time();

      GPUmem.deallocate(dev_dtotal, total_dsize);
      auto t4 = tools::get_time();

      /*	
         std::cout << "GEMM[grouped]: talloc=" << tools::get_duration(t1-t0)
         << " t2gpu=" << tools::get_duration(t2-t1)
         << " tcomp=" << tools::get_duration(t3-t2)
         << " tdealloc=" << tools::get_duration(t4-t3)
         << " total=" << tools::get_duration(t4-t0)
         << " flops=" << cost/tools::get_duration(t3-t2)
         << std::endl;
         */
   }

#else

   // double
   inline void xgemm_batch_gpu_grouped(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const double *alpha, const double **a_array, const MKL_INT *lda_array, 
         const double **b_array, const MKL_INT *ldb_array,
         const double *beta, double **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_grouped"<<std::endl;
         return ;
      }

      //auto t0 = tools::get_time();

      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      //auto t1 = tools::get_time();

      double** dev_a_array = (double**)dev_dtotal;
      double** dev_b_array = dev_a_array + batch_count;
      double** dev_c_array = dev_b_array + batch_count;
      GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));

      //auto t2 = tools::get_time();

      double cost = 0.0; 
      for(int i=0; i<gsta.size()-1; i++){
         int ista = gsta[i];
         int nbatch = gsta[i+1]-ista;
         // convert from MKL_INT to int 
         int m = m_array[ista], n = n_array[ista], k = k_array[ista];
         int lda = lda_array[ista], ldb = ldb_array[ista], ldc = ldc_array[ista]; 
         cublasOperation_t transA = CUBLAS_OP_N ;
         if(transa=='T' || transa=='C'){
            transA = CUBLAS_OP_T;
         }
         cublasOperation_t transB = CUBLAS_OP_N ;
         if(transb=='T' || transb=='C'){
            transB = CUBLAS_OP_T;
         }
         cost += double(m)*n*k*2*nbatch;
         // https://docs.nvidia.com/cuda/cublas/index.html
         //auto ti = tools::get_time();
         CUBLAS_CHECK(cublasDgemmBatched(handle_cublas,
                  transA, transB,
                  m, n, k,
                  alpha,
                  &dev_a_array[ista], lda, // pointer of matrix should be on device
                  &dev_b_array[ista], ldb,
                  beta,
                  &dev_c_array[ista], ldc,
                  nbatch));
         /*
            auto tj = tools::get_time();
            GPUmem.sync();
            auto tf = tools::get_time();
            std::cout << "i=" << i 
            << " m,n,k,b=" << m << "," << n << "," << k << "," << nbatch
            << " cost=" << double(m)*n*k*2*nbatch
            << " flops=" << double(m)*n*k*2*nbatch/tools::get_duration(tf-ti)
            << " t0=" << tools::get_duration(tj-ti)
            << " t1=" << tools::get_duration(tf-tj)
            << std::endl;
            */
         /*
            for(int j=0; j<nbatch; j++){
            cublasDgemm(handle_cublas,
            transA, transB,
            m, n, k,
            alpha,
            a_array[ista+j], lda,
            b_array[ista+j], ldb,
            beta,
            c_array[ista+j], ldc);
            }
            */
      } // group
      //auto t3 = tools::get_time();

      GPUmem.deallocate(dev_dtotal, total_dsize);
      //auto t4 = tools::get_time();

      /*
         std::cout << "GEMM[grouped]: talloc=" << tools::get_duration(t1-t0)
         << " t2gpu=" << tools::get_duration(t2-t1)
         << " tcomp=" << tools::get_duration(t3-t2)
         << " tdealloc=" << tools::get_duration(t4-t3)
         << " total=" << tools::get_duration(t4-t0)
         << " flops=" << cost/tools::get_duration(t3-t2)
         << std::endl;
         */
   }

#endif // GEMMGROUPED [for cublas>=12.4]

   // complex
   inline void xgemm_batch_gpu_grouped(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **a_array, const MKL_INT *lda_array,
         const std::complex<double> **b_array, const MKL_INT *ldb_array,
         const std::complex<double> *beta, std::complex<double> **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_grouped"<<std::endl;
         return ;
      }
      std::cout << "COMPLEX CASE IS NOT IMPLEMENTED YET!" << std::endl;
      exit(1);
   }

   // --- GEMM STREAM ---
   // double
   inline void xgemm_batch_gpu_stream(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const double *alpha, const double **a_array, const MKL_INT *lda_array, 
         const double **b_array, const MKL_INT *ldb_array,
         const double *beta, double **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_stream"<<std::endl;
         return ;
      }

      //auto t0 = tools::get_time();

      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_dtotal = GPUmem.allocate(total_dsize);

      //auto t1 = tools::get_time();

      double** dev_a_array = (double**)dev_dtotal;
      double** dev_b_array = dev_a_array + batch_count;
      double** dev_c_array = dev_b_array + batch_count;
      GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
      GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));

      //auto t2 = tools::get_time();

      size_t gsize = gsta.size()-1;
      int ntimes = (gsize+NSTREAMS-1)/NSTREAMS; 
      for(int k=0; k<ntimes; k++){
         size_t off = k*NSTREAMS;
         size_t jlen = std::min(gsize-off, size_t(NSTREAMS));

         for(int j=0; j<jlen; j++){
            size_t jdx = off+j;
            CUBLAS_CHECK(cublasSetStream(handle_cublas, custream[j])); 

            int ista = gsta[jdx];
            int nbatch = gsta[jdx+1]-ista;
            // convert from MKL_INT to int 
            int m = m_array[ista], n = n_array[ista], k = k_array[ista];
            int lda = lda_array[ista], ldb = ldb_array[ista], ldc = ldc_array[ista]; 
            cublasOperation_t transA = CUBLAS_OP_N ;
            if(transa=='T' || transa=='C'){
               transA = CUBLAS_OP_T;
            }
            cublasOperation_t transB = CUBLAS_OP_N ;
            if(transb=='T' || transb=='C'){
               transB = CUBLAS_OP_T;
            }
            // https://docs.nvidia.com/cuda/cublas/index.html
            CUBLAS_CHECK(cublasDgemmBatched(handle_cublas,
                     transA, transB,
                     m, n, k,
                     alpha,
                     &dev_a_array[ista], lda, // pointer of matrix should be on device
                     &dev_b_array[ista], ldb,
                     beta,
                     &dev_c_array[ista], ldc,
                     nbatch));
         } // j

         for(int j=0; j<jlen; j++){
            CUDA_CHECK(cudaStreamSynchronize(custream[j]));
         }
      } // k

      //auto t3 = tools::get_time();

      GPUmem.deallocate(dev_dtotal, total_dsize);

      /*  
          auto t4 = tools::get_time();
          std::cout << "GEMM[stream]: talloc=" << tools::get_duration(t1-t0)
          << " t2gpu=" << tools::get_duration(t2-t1)
          << " tcomp=" << tools::get_duration(t3-t2)
          << " tdealloc=" << tools::get_duration(t4-t3)
          << " total=" << tools::get_duration(t4-t0)
          << std::endl;
          */
   }

   // complex
   inline void xgemm_batch_gpu_stream(const char transa, const char transb, 
         const MKL_INT *m_array, const MKL_INT *n_array, const MKL_INT *k_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **a_array, const MKL_INT *lda_array,
         const std::complex<double> **b_array, const MKL_INT *ldb_array,
         const std::complex<double> *beta, std::complex<double> **c_array, const MKL_INT *ldc_array, 
         const MKL_INT batch_count,
         const std::vector<int>& gsta){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_stream"<<std::endl;
         return ;
      }
      std::cout << "COMPLEX CASE IS NOT IMPLEMENTED YET!" << std::endl;
      exit(1);
   }

} // linalg

#endif

#endif//GPU
