#ifdef GPU

#ifndef GPU_BLAS_BATCH_H
#define GPU_BLAS_BATCH_H

#include "gpu_env.h"

// wrapper for BLAS
namespace linalg{

// double
inline void xgemm_batch_gpu(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta, double **c_array, const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
    if(batch_count <= 0)
    {
        std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu"<<std::endl;
        return ;
    }
#ifdef USE_HIP
    int* dev_total;
    double** dev_total_addr;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    size_t total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);


    size_t total_size_addr =0 ;
    //dev_array_ptr
    total_size_addr +=3*batch_count*sizeof(double*);


	 HIP_CHECK(hipMalloc((void**)&dev_total, total_size));
	 HIP_CHECK(hipMemset((void*)dev_total, 0, total_size));

	 HIP_CHECK(hipMalloc((void**)&dev_total_addr, total_size_addr));
	 HIP_CHECK(hipMemset((void*)dev_total_addr, 0, total_size_addr));


    dev_m = (int*)dev_total;
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)dev_total_addr ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    HIP_CHECK(hipMemcpy(dev_m, m_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_n, n_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_k, k_array, batch_count*sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(dev_lda, lda_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), hipMemcpyHostToDevice));


    HIP_CHECK(hipMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), hipMemcpyHostToDevice));

    //magma_init();
    //magma_queue_t queue=0;
    //magma_setdevice(0);
    //magma_queue_create(0,&queue);

    magma_trans_t transA =  MagmaNoTrans ;
    magma_trans_t transB =  MagmaNoTrans ;


    if(transa=='T')
    {
        transA = MagmaTrans;
    }else if (transa == 'C'){
        transA = MagmaConjTrans;
    }

    if(transb=='T')
    {
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

    HIP_CHECK(hipFree(dev_total));     
    HIP_CHECK(hipFree(dev_total_addr));     
#else
#if defined(USE_CUDA_OPERATION)
    int* dev_total;
    double** dev_total_addr;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    size_t total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);


    size_t total_size_addr =0 ;
    //dev_array_ptr
    total_size_addr +=3*batch_count*sizeof(double*);


	 CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));
	 CUDA_CHECK(cudaMemset((void*)dev_total, 0, total_size));

	 CUDA_CHECK(cudaMalloc((void**)&dev_total_addr, total_size_addr));
	 CUDA_CHECK(cudaMemset((void*)dev_total_addr, 0, total_size_addr));


    dev_m = (int*)dev_total;
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)dev_total_addr ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));

    //magma_init();
    //magma_queue_t queue=0;
    //magma_setdevice(0);
    //magma_queue_create(0,&queue);

    magma_trans_t transA =  MagmaNoTrans ;
    magma_trans_t transB =  MagmaNoTrans ;


    if(transa=='T')
    {
        transA = MagmaTrans;
    }else if (transa == 'C'){
        transA = MagmaConjTrans;
    }

    if(transb=='T')
    {
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

    CUDA_CHECK(cudaFree(dev_total));     
    CUDA_CHECK(cudaFree(dev_total_addr));     

#else

/*
    // lzd: copied from above
    double* dev_total;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    size_t total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(double*), cudaMemcpyHostToDevice));

    //magma_init();
    //magma_queue_t queue=0;
    //magma_setdevice(0);
    //magma_queue_create(0,&queue);

    magma_trans_t transA =  MagmaNoTrans ;
    magma_trans_t transB =  MagmaNoTrans ;


    if(transa=='T')
    {
        transA = MagmaTrans;
    }else if (transa == 'C'){
        transA = MagmaConjTrans;
    }

    if(transb=='T')
    {
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

    CUDA_CHECK(cudaFree(dev_total));    
*/
 

    magma_ptr dev_total;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    size_t total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size += 3*batch_count*sizeof(double*);

    //CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));
    MAGMA_CHECK(magma_malloc(&dev_total, total_size));

    // lzd debug
    cudaMemset((void**)&dev_total, 0, total_size);

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;
/*
    // lzd
    magma_int_t* dev_m;
    magma_int_t* dev_n;
    magma_int_t* dev_k;
    magma_int_t* dev_lda;
    magma_int_t* dev_ldb;
    magma_int_t* dev_ldc;

    MAGMA_CHECK(magma_imalloc(&dev_m, 6*(batch_count+1)));
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);
    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    double** dev_a_array_ptr;
    double** dev_b_array_ptr;
    double** dev_c_array_ptr;
    MAGMA_CHECK(magma_dmalloc((double **)(&dev_a_array_ptr), 3*batch_count));
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;
*/

    // copy
    magma_isetvector(batch_count, m_array, 1, dev_m, 1, magma_queue);
    magma_isetvector(batch_count, n_array, 1, dev_n, 1, magma_queue);
    magma_isetvector(batch_count, k_array, 1, dev_k, 1, magma_queue);

    magma_isetvector(batch_count, lda_array, 1, dev_lda, 1, magma_queue);
    magma_isetvector(batch_count, ldb_array, 1, dev_ldb, 1, magma_queue);
    magma_isetvector(batch_count, ldc_array, 1, dev_ldc, 1, magma_queue);

    magma_setvector(batch_count, sizeof(double*),  a_array, 1, dev_a_array_ptr, 1, magma_queue);
    magma_setvector(batch_count, sizeof(double*),  b_array, 1, dev_b_array_ptr, 1, magma_queue);
    magma_setvector(batch_count, sizeof(double*),  c_array, 1, dev_c_array_ptr, 1, magma_queue);

    //magma_init();
    //magma_queue_t queue=0;
    //magma_setdevice(0);
    //magma_queue_create(0,&queue);

    magma_trans_t transA =  MagmaNoTrans ;
    magma_trans_t transB =  MagmaNoTrans ;


    if(transa=='T')
    {
        transA = MagmaTrans;
    }else if (transa == 'C'){
        transA = MagmaConjTrans;
    }

    if(transb=='T')
    {
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

    MAGMA_CHECK(magma_free(dev_total));
/*
    MAGMA_CHECK(magma_free(dev_m));
    MAGMA_CHECK(magma_free(dev_a_array_ptr));
*/
#endif
#endif//USE_HIP

}

// complex
inline void xgemm_batch_gpu(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const std::complex<double> *alpha, 
        const std::complex<double> **a_array, const int *lda_array,
        const std::complex<double> **b_array, const int *ldb_array,
        const std::complex<double> *beta, std::complex<double> **c_array, 
        const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
    if(batch_count <= 0)
    {
        std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu"<<std::endl;
        return ;
    }
#if 0
    magmaDoubleComplex* dev_total;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    magmaDoubleComplex ** dev_a_array_ptr;
    magmaDoubleComplex ** dev_b_array_ptr;
    magmaDoubleComplex ** dev_c_array_ptr;


    size_t total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(magmaDoubleComplex*);

    CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (magmaDoubleComplex**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_b_array_ptr, b_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_c_array_ptr, c_array, batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));

    //magma_init();
    //magma_queue_t queue=0;
    //magma_setdevice(0);
    //magma_queue_create(0,&queue);

    magma_trans_t transA =  MagmaNoTrans ;
    magma_trans_t transB =  MagmaNoTrans ;


    if(transa=='T')
    {
        transA = MagmaTrans;
    }else if (transa == 'C'){
        transA = MagmaConjTrans;
    }

    if(transb=='T')
    {
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

    CUDA_CHECK(cudaFree(dev_total));     

#endif
}

} // linalg

#endif

#endif//GPU
