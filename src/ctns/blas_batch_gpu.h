
#ifdef GPU

#ifndef BLAS_BATCH_GPU_H
#define BLAS_BATCH_GPU_H

#include "../gpu/gpu_env.h"

// wrapper for BLAS
namespace linalg{

inline void xgemm_batch_gpu(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta, double **c_array, const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
    if(batch_count <= 0)
    {
        std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_precopy"<<std::endl;
        return ;
    }
#ifdef USE_HIP
    double* dev_total;

    double* dev_a;
    double* dev_b;
    double* dev_c;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    int total_size = (a_total_count + b_total_count + c_total_count)*sizeof(double);
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    HIP_CHECK(hipMalloc((void**)&dev_total, total_size));

    dev_a = dev_total + 0;
    dev_b = dev_a + a_total_count;
    dev_c = dev_b + b_total_count;

    dev_m = (int*)((void*)dev_c + c_total_count*sizeof(double));
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    for(int i=0;i<batch_count;i++)
    {
        HIP_CHECK(hipMemcpy(dev_a, a_array[i], m_array[i]*k_array[i]*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dev_b, b_array[i], k_array[i]*n_array[i]*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dev_c, c_array[i], m_array[i]*n_array[i]*sizeof(double), hipMemcpyHostToDevice));

        dev_a +=m_array[i]*k_array[i];
        dev_b +=k_array[i]*n_array[i];
        dev_c +=m_array[i]*n_array[i];
    }

    HIP_CHECK(hipMemcpy(dev_m, m_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_n, n_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_k, k_array, batch_count*sizeof(int), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemcpy(dev_lda, lda_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), hipMemcpyHostToDevice));

    double ** a_array_ptr = (double**)malloc(3*batch_count*sizeof(double*));
    double ** b_array_ptr = a_array_ptr + batch_count;
    double ** c_array_ptr = b_array_ptr + batch_count;

    a_array_ptr[0] = (double*)dev_total + 0;
    b_array_ptr[0] = (double*)dev_total + a_total_count; 
    c_array_ptr[0] = (double*)dev_total + a_total_count + b_total_count; 

    for(int i=1;i<batch_count;i++)
    {
        a_array_ptr[i] = a_array_ptr[i-1] + m_array[i-1]*k_array[i-1];
        b_array_ptr[i] = b_array_ptr[i-1] + k_array[i-1]*n_array[i-1];
        c_array_ptr[i] = c_array_ptr[i-1] + m_array[i-1]*n_array[i-1];
    }

    HIP_CHECK(hipMemcpy(dev_a_array_ptr, a_array_ptr, 3*batch_count*sizeof(double*), hipMemcpyHostToDevice));

    magma_queue_t queue=::magma_queue;
    /**
    int device=0;
    magma_queue_t queue=0;
    magma_init();
    magma_getdevice(&device);
    magma_queue_create(device,&queue);
    **/

    

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
            queue
            );

    dev_c = dev_total + a_total_count + b_total_count;
    for(int i=0;i<batch_count;i++)
    {
        HIP_CHECK(hipMemcpy(c_array[i], dev_c, m_array[i]*n_array[i]*sizeof(double), hipMemcpyDeviceToHost));
        dev_c +=m_array[i]*n_array[i];
    }

    if(a_array_ptr)
        free(a_array_ptr);
    HIP_CHECK(hipFree(dev_total));     

#else
#if defined(USE_CUDA_OPERATION)

    double* dev_total;

    double* dev_a;
    double* dev_b;
    double* dev_c;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    double ** dev_a_array_ptr;
    double ** dev_b_array_ptr;
    double ** dev_c_array_ptr;


    int total_size = (a_total_count + b_total_count + c_total_count)*sizeof(double);
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));

    dev_a = dev_total + 0;
    dev_b = dev_a + a_total_count;
    dev_c = dev_b + b_total_count;

    dev_m = (int*)((void*)dev_c + c_total_count*sizeof(double));
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


    for(int i=0;i<batch_count;i++)
    {
        CUDA_CHECK(cudaMemcpy(dev_a, a_array[i], m_array[i]*k_array[i]*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, b_array[i], k_array[i]*n_array[i]*sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_c, c_array[i], m_array[i]*n_array[i]*sizeof(double), cudaMemcpyHostToDevice));

        dev_a +=m_array[i]*k_array[i];
        dev_b +=k_array[i]*n_array[i];
        dev_c +=m_array[i]*n_array[i];
    }

    CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    double ** a_array_ptr = (double**)malloc(3*batch_count*sizeof(double*));
    double ** b_array_ptr = a_array_ptr + batch_count;
    double ** c_array_ptr = b_array_ptr + batch_count;

    a_array_ptr[0] = (double*)dev_total + 0;
    b_array_ptr[0] = (double*)dev_total + a_total_count; 
    c_array_ptr[0] = (double*)dev_total + a_total_count + b_total_count; 

    for(int i=1;i<batch_count;i++)
    {
        a_array_ptr[i] = a_array_ptr[i-1] + m_array[i-1]*k_array[i-1];
        b_array_ptr[i] = b_array_ptr[i-1] + k_array[i-1]*n_array[i-1];
        c_array_ptr[i] = c_array_ptr[i-1] + m_array[i-1]*n_array[i-1];
    }

    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array_ptr, 3*batch_count*sizeof(double*), cudaMemcpyHostToDevice));

    magma_queue_t queue=::magma_queue;
    /**
    int device=0;
    magma_queue_t queue=0;
    magma_init();
    magma_getdevice(&device);
    magma_queue_create(device,&queue);
    **/

    

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
            queue
            );

    dev_c = dev_total + a_total_count + b_total_count;
    for(int i=0;i<batch_count;i++)
    {
        CUDA_CHECK(cudaMemcpy(c_array[i], dev_c, m_array[i]*n_array[i]*sizeof(double), cudaMemcpyDeviceToHost));
        dev_c +=m_array[i]*n_array[i];
    }

    if(a_array_ptr)
        free(a_array_ptr);
    CUDA_CHECK(cudaFree(dev_total));     

    /**
    magma_queue_destroy(queue);
    magma_finalize();
    **/
#else
    double* dev_total;
    double* dev_a;
    double* dev_b;
    double* dev_c;

    double** dev_a_array_ptr;
    double** dev_b_array_ptr;
    double** dev_c_array_ptr;

    magma_int_t* dev_m;
    magma_int_t* dev_n;
    magma_int_t* dev_k;

    magma_int_t* dev_lda;
    magma_int_t* dev_ldb;
    magma_int_t* dev_ldc;



    MAGMA_CHECK(magma_dmalloc(&dev_total, a_total_count + b_total_count + c_total_count));
    dev_a = dev_total + 0;
    dev_b = dev_a + a_total_count;
    dev_c = dev_b + b_total_count;

    MAGMA_CHECK(magma_imalloc(&dev_m, 6*(batch_count+1)));
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    CUDA_CHECK(cudaMalloc(&dev_a_array_ptr, 3*batch_count*sizeof(double*)));
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;

    for(int i=0;i<batch_count;i++)
    {

        magma_dsetmatrix(m_array[i], k_array[i], a_array[i], m_array[i], dev_a, m_array[i], magma_queue);
        magma_dsetmatrix(k_array[i], n_array[i], b_array[i], k_array[i], dev_b, k_array[i], magma_queue);
        magma_dsetmatrix(m_array[i], n_array[i], c_array[i], m_array[i], dev_c, m_array[i], magma_queue);

        dev_a +=m_array[i]*k_array[i];
        dev_b +=k_array[i]*n_array[i];
        dev_c +=m_array[i]*n_array[i];
    }

    magma_isetvector(batch_count, m_array, 1, dev_m, 1, magma_queue);
    magma_isetvector(batch_count, n_array, 1, dev_n, 1, magma_queue);
    magma_isetvector(batch_count, k_array, 1, dev_k, 1, magma_queue);

    magma_isetvector(batch_count, lda_array, 1, dev_lda, 1, magma_queue);
    magma_isetvector(batch_count, ldb_array, 1, dev_ldb, 1, magma_queue);
    magma_isetvector(batch_count, ldc_array, 1, dev_ldc, 1, magma_queue);

    double ** a_array_ptr = (double**)malloc(3*batch_count*sizeof(double*));
    if(!a_array_ptr)
    {
        exit(1);
    
    }
    double ** b_array_ptr = a_array_ptr + batch_count;
    double ** c_array_ptr = b_array_ptr + batch_count;

    a_array_ptr[0] = (double*)dev_total + 0;
    b_array_ptr[0] = (double*)dev_total + a_total_count; 
    c_array_ptr[0] = (double*)dev_total + a_total_count + b_total_count; 


    for(int i=1;i<batch_count;i++)
    {
        a_array_ptr[i] = a_array_ptr[i-1] + m_array[i-1]*k_array[i-1];
        b_array_ptr[i] = b_array_ptr[i-1] + k_array[i-1]*n_array[i-1];
        c_array_ptr[i] = c_array_ptr[i-1] + m_array[i-1]*n_array[i-1];
    }
    //magma_setvector(3*batch_count, sizeof(double*),  a_array_ptr, 1, dev_a_array_ptr, 1, magma_queue);
    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array_ptr, 3*batch_count*sizeof(double*), cudaMemcpyHostToDevice));


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

    dev_c = dev_total + a_total_count + b_total_count;
    for(int i=0;i<batch_count;i++)
    {
        magma_dgetmatrix(m_array[i], n_array[i], dev_c, m_array[i], c_array[i], m_array[i], magma_queue);
        dev_c +=m_array[i]*n_array[i];
    }

    if(a_array_ptr)
        free(a_array_ptr);
    MAGMA_CHECK(magma_free(dev_m));
    CUDA_CHECK(cudaFree(dev_a_array_ptr));
    MAGMA_CHECK(magma_free(dev_total));

#endif
#endif //USE_HIP
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
#if 0
    //magmaDoubleComplex* dev_total;
    magmaDoubleComplex* dev_total;

    magmaDoubleComplex* dev_a;
    magmaDoubleComplex* dev_b;
    magmaDoubleComplex* dev_c;

    int  *dev_m;
    int  *dev_n;
    int  *dev_k;

    int  *dev_lda;
    int  *dev_ldb;
    int  *dev_ldc;

    magmaDoubleComplex ** dev_a_array_ptr;
    magmaDoubleComplex ** dev_b_array_ptr;
    magmaDoubleComplex ** dev_c_array_ptr;


    int total_size = (a_total_count + b_total_count + c_total_count)*sizeof(magmaDoubleComplex);
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(magmaDoubleComplex*);

    CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));

    dev_a = dev_total + 0;
    dev_b = dev_a + a_total_count;
    dev_c = dev_b + b_total_count;

    dev_m = (int*)((void*)dev_c + c_total_count*sizeof(magmaDoubleComplex));
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (magmaDoubleComplex**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;

    for(int i=0;i<batch_count;i++)
    {
        CUDA_CHECK(cudaMemcpy(dev_a, a_array[i], m_array[i]*k_array[i]*sizeof(magmaDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_b, b_array[i], k_array[i]*n_array[i]*sizeof(magmaDoubleComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dev_c, c_array[i], m_array[i]*n_array[i]*sizeof(magmaDoubleComplex), cudaMemcpyHostToDevice));

        dev_a +=m_array[i]*k_array[i];
        dev_b +=k_array[i]*n_array[i];
        dev_c +=m_array[i]*n_array[i];
    }

    CUDA_CHECK(cudaMemcpy(dev_m, m_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_n, n_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_k, k_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dev_lda, lda_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldb, ldb_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_ldc, ldc_array, batch_count*sizeof(int), cudaMemcpyHostToDevice));

    magmaDoubleComplex ** a_array_ptr = (magmaDoubleComplex**)malloc(3*batch_count*sizeof(magmaDoubleComplex*));
    magmaDoubleComplex ** b_array_ptr = a_array_ptr + batch_count;
    magmaDoubleComplex ** c_array_ptr = b_array_ptr + batch_count;

    a_array_ptr[0] = (magmaDoubleComplex*)dev_a + 0;
    b_array_ptr[0] = (magmaDoubleComplex*)dev_a + a_total_count; 
    c_array_ptr[0] = (magmaDoubleComplex*)dev_a + a_total_count + b_total_count; 

    for(int i=1;i<batch_count;i++)
    {
        a_array_ptr[i] = a_array_ptr[i-1] + m_array[i-1]*k_array[i-1];
        b_array_ptr[i] = b_array_ptr[i-1] + k_array[i-1]*n_array[i-1];
        c_array_ptr[i] = c_array_ptr[i-1] + m_array[i-1]*n_array[i-1];
    }

    CUDA_CHECK(cudaMemcpy(dev_a_array_ptr, a_array_ptr, 3*batch_count*sizeof(magmaDoubleComplex*), cudaMemcpyHostToDevice));

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

    dev_c = dev_total + a_total_count + b_total_count;
    for(int i=0;i<batch_count;i++)
    {
        CUDA_CHECK(cudaMemcpy(c_array[i], dev_c, m_array[i]*n_array[i]*sizeof(magmaDoubleComplex), cudaMemcpyDeviceToHost));
        dev_c +=m_array[i]*n_array[i];
    }

    if(a_array_ptr)
        free(a_array_ptr);
    CUDA_CHECK(cudaFree(dev_total));     


#endif
}


inline void xgemm_batch_gpu_precopy(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta, double **c_array, const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
    if(batch_count <= 0)
    {
        std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_precopy"<<std::endl;
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


    int total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    //CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));
    MAGMA_CHECK(magma_malloc(&dev_total, total_size));

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


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

#endif
#endif//USE_HIP

}

inline void xgemm_batch_gpu_precopy_stream(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta, double **c_array, const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
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


    int total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    //CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));
    MAGMA_CHECK(magma_malloc(&dev_total, total_size));

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


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
}

inline void xgemm_batch_gpu_precopy_new(const char transa, const char transb, 
        const int *m_array, const int *n_array, const int *k_array,
        const double *alpha, const double **a_array, const int *lda_array, 
        const double **b_array, const int *ldb_array,
        const double *beta, double **c_array, const int *ldc_array, 
        const int batch_count, const int a_total_count, const int b_total_count, const int c_total_count)
{
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


    int total_size =0 ;
    //dev_m,dev_n,dev_k,dev_lda,dev_ldb,dev_ldc
    total_size += 6*(batch_count+1)*sizeof(int);
    //dev_array_ptr
    total_size +=3*batch_count*sizeof(double*);

    //CUDA_CHECK(cudaMalloc((void**)&dev_total, total_size));
    MAGMA_CHECK(magma_malloc(&dev_total, total_size));

    dev_m = (int*)((void*)dev_total);
    dev_n = dev_m + (batch_count+1);
    dev_k = dev_n + (batch_count+1);

    dev_lda = dev_k + (batch_count+1);
    dev_ldb = dev_lda + (batch_count+1);
    dev_ldc = dev_ldb + (batch_count+1);

    dev_a_array_ptr= (double**)((void*)dev_ldc +(batch_count+1)*sizeof(int)) ;
    dev_b_array_ptr = dev_a_array_ptr + batch_count;
    dev_c_array_ptr = dev_b_array_ptr + batch_count;


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
}

// complex
inline void xgemm_batch_gpu_precopy(const char transa, const char transb, 
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
        std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_precopy"<<std::endl;
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


    int total_size =0 ;
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
