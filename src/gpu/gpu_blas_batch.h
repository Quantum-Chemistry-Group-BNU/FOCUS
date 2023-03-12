#ifdef GPU

#ifndef GPU_BLAS_BATCH_H
#define GPU_BLAS_BATCH_H

#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

/*
 * TODO: gemv
 *
   // double
   inline void xgemv_batch_gpu(const char transa, const char transb, 
         const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
         const double *alpha, const double **a_array, const magma_int_t *lda_array, 
         const double **b_array, const magma_int_t *ldb_array,
         const double *beta, double **c_array, const magma_int_t *ldc_array, 
         const magma_int_t batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
      size_t total_isize = 6*(batch_count+1)*sizeof(magma_int_t);
      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);
      magma_int_t* dev_m = (magma_int_t*)dev_itotal;
      magma_int_t* dev_n = dev_m + (batch_count+1);
      magma_int_t* dev_k = dev_n + (batch_count+1);
      magma_int_t* dev_lda = dev_k + (batch_count+1);
      magma_int_t* dev_ldb = dev_lda + (batch_count+1);
      magma_int_t* dev_ldc = dev_ldb + (batch_count+1);
      double** dev_a_array_ptr= (double**)dev_dtotal;
      double** dev_b_array_ptr = dev_a_array_ptr + batch_count;
      double** dev_c_array_ptr = dev_b_array_ptr + batch_count;

#ifdef USE_HIP
      HIP_CHECK(hipMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
#else
      CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
#endif

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
            dev_a_array_ptr,
            dev_lda,
            dev_b_array_ptr,
            dev_ldb,
            beta[0],
            dev_c_array_ptr,
            dev_ldc,
            batch_count,
            magma_queue
            );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }
*/

   // double
   inline void xgemm_batch_gpu(const char transa, const char transb, 
         const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
         const double *alpha, const double **a_array, const magma_int_t *lda_array, 
         const double **b_array, const magma_int_t *ldb_array,
         const double *beta, double **c_array, const magma_int_t *ldc_array, 
         const magma_int_t batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
      size_t total_isize = 6*(batch_count+1)*sizeof(magma_int_t);
      size_t total_dsize = 3*batch_count*sizeof(double*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);
      magma_int_t* dev_m = (magma_int_t*)dev_itotal;
      magma_int_t* dev_n = dev_m + (batch_count+1);
      magma_int_t* dev_k = dev_n + (batch_count+1);
      magma_int_t* dev_lda = dev_k + (batch_count+1);
      magma_int_t* dev_ldb = dev_lda + (batch_count+1);
      magma_int_t* dev_ldc = dev_ldb + (batch_count+1);
      double** dev_a_array_ptr= (double**)dev_dtotal;
      double** dev_b_array_ptr = dev_a_array_ptr + batch_count;
      double** dev_c_array_ptr = dev_b_array_ptr + batch_count;

#ifdef USE_HIP
      HIP_CHECK(hipMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
#else
      CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
#endif

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
            dev_a_array_ptr,
            dev_lda,
            dev_b_array_ptr,
            dev_ldb,
            beta[0],
            dev_c_array_ptr,
            dev_ldc,
            batch_count,
            magma_queue
            );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }

   // complex
   inline void xgemm_batch_gpu(const char transa, const char transb, 
         const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
         const std::complex<double> *alpha, 
         const std::complex<double> **a_array, const magma_int_t *lda_array,
         const std::complex<double> **b_array, const magma_int_t *ldb_array,
         const std::complex<double> *beta, std::complex<double> **c_array, const magma_int_t *ldc_array, 
         const magma_int_t batch_count){

      if(batch_count <= 0){
         std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu"<<std::endl;
         return ;
      }

      //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
      size_t total_isize = 6*(batch_count+1)*sizeof(magma_int_t);
      size_t total_dsize = 3*batch_count*sizeof(magmaDoubleComplex*);
      void* dev_itotal = GPUmem.allocate(total_isize);
      void* dev_dtotal = GPUmem.allocate(total_dsize);
      magma_int_t* dev_m = (magma_int_t*)dev_itotal;
      magma_int_t* dev_n = dev_m + (batch_count+1);
      magma_int_t* dev_k = dev_n + (batch_count+1);
      magma_int_t* dev_lda = dev_k + (batch_count+1);
      magma_int_t* dev_ldb = dev_lda + (batch_count+1);
      magma_int_t* dev_ldc = dev_ldb + (batch_count+1);
      magmaDoubleComplex** dev_a_array_ptr= (magmaDoubleComplex**)dev_dtotal;
      magmaDoubleComplex** dev_b_array_ptr = dev_a_array_ptr + batch_count;
      magmaDoubleComplex** dev_c_array_ptr = dev_b_array_ptr + batch_count;

#ifdef USE_HIP
      HIP_CHECK(hipMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(magmaDoubleComplex*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(magmaDoubleComplex*), hipMemcpyHostToDevice));
      HIP_CHECK(hipMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(magmaDoubleComplex*), hipMemcpyHostToDevice));
#else
      CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));
#endif

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
            dev_a_array_ptr,
            dev_lda,
            dev_b_array_ptr,
            dev_ldb,
            beta1,
            dev_c_array_ptr,
            dev_ldc,
            batch_count,
            magma_queue
            );

      GPUmem.deallocate(dev_dtotal, total_dsize);
      GPUmem.deallocate(dev_itotal, total_isize);
   }

} // linalg

#endif

#endif//GPU
