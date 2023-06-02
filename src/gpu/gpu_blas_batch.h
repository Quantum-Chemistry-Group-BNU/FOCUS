#ifdef GPU

#ifndef GPU_BLAS_BATCH_H
#define GPU_BLAS_BATCH_H

#include "gpu_env.h"

// wrapper for BLAS_BATCH
namespace linalg{

    // --- GEMV ---
    // double
    inline void xgemv_batch_gpu_magma(const char trans,
            const magma_int_t *m_array, const magma_int_t *n_array,
            const double *alpha, const double **A_array, const magma_int_t *lda_array, 
            const double **X_array, const magma_int_t *incx_array,
            const double *beta, double** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_magma"<<std::endl;
            return ;
        }

        //dev_m,dev_n,dev_lda,dev_incx,dev_incy
        size_t total_isize = 5*(batch_count+1)*sizeof(magma_int_t);
        size_t total_dsize = 3*batch_count*sizeof(double*);
        void* dev_itotal = GPUmem.allocate(total_isize);
        void* dev_dtotal = GPUmem.allocate(total_dsize);

        magma_int_t* dev_m = (magma_int_t*)dev_itotal;
        magma_int_t* dev_n = dev_m + (batch_count+1);
        magma_int_t* dev_lda = dev_n + (batch_count+1);
        magma_int_t* dev_incx = dev_lda + (batch_count+1);
        magma_int_t* dev_incy = dev_incx + (batch_count+1);
        double** dev_A_array= (double**)dev_dtotal;
        double** dev_X_array = dev_A_array + batch_count;
        double** dev_Y_array = dev_X_array + batch_count;

        GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_incx, incx_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_incy, incy_array, batch_count*sizeof(magma_int_t));
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
            const magma_int_t *m_array, const magma_int_t *n_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **A_array, const magma_int_t *lda_array, 
            const std::complex<double> **X_array, const magma_int_t *incx_array,
            const std::complex<double> *beta, std::complex<double>** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_magma"<<std::endl;
            return ;
        }

        //dev_m,dev_n,dev_lda,dev_incx,dev_incy
        size_t total_isize = 5*(batch_count+1)*sizeof(magma_int_t);
        size_t total_dsize = 3*batch_count*sizeof(magmaDoubleComplex*);
        void* dev_itotal = GPUmem.allocate(total_isize);
        void* dev_dtotal = GPUmem.allocate(total_dsize);

        magma_int_t* dev_m = (magma_int_t*)dev_itotal;
        magma_int_t* dev_n = dev_m + (batch_count+1);
        magma_int_t* dev_lda = dev_n + (batch_count+1);
        magma_int_t* dev_incx = dev_lda + (batch_count+1);
        magma_int_t* dev_incy = dev_incx + (batch_count+1);
        magmaDoubleComplex** dev_A_array= (magmaDoubleComplex**)dev_dtotal;
        magmaDoubleComplex** dev_X_array = dev_A_array + batch_count;
        magmaDoubleComplex** dev_Y_array = dev_X_array + batch_count;

        GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_incx, incx_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_incy, incy_array, batch_count*sizeof(magma_int_t));
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

#ifndef HIP
    // --- GEMV GROUPED ---
    // double
    inline void xgemv_batch_gpu_grouped(const char trans,
            const magma_int_t *m_array, const magma_int_t *n_array,
            const double *alpha, const double **A_array, const magma_int_t *lda_array, 
            const double **X_array, const magma_int_t *incx_array,
            const double *beta, double** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count,
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
           // convert from magma_int_t to int 
           int m = m_array[ista], n = n_array[ista];
           int lda = lda_array[ista], incx = incx_array[ista], incy = incy_array[ista]; 
           cublasOperation_t Trans = CUBLAS_OP_N ;
           if(trans=='T' || trans=='C'){
              Trans = CUBLAS_OP_T;
           }
           // https://docs.nvidia.com/cuda/cublas/index.html
           CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
                              Trans, 
                              m, n,
                              alpha,
                              &dev_A_array[ista], lda, // pointer should be on device
                              &dev_X_array[ista], incx,
                              beta,
                              &dev_Y_array[ista], incy,
                              nbatch));
           /*
           for(int j=0; j<nbatch; j++){
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
            const magma_int_t *m_array, const magma_int_t *n_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **A_array, const magma_int_t *lda_array, 
            const std::complex<double> **X_array, const magma_int_t *incx_array,
            const std::complex<double> *beta, std::complex<double>** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_grouped"<<std::endl;
            return ;
        }
        std::cout << "NOT IMPLEMENTED YET!" << std::endl;
        exit(1);
    }

    // --- GEMV STREAM ---
    // double
    inline void xgemv_batch_gpu_stream(const char trans,
            const magma_int_t *m_array, const magma_int_t *n_array,
            const double *alpha, const double **A_array, const magma_int_t *lda_array, 
            const double **X_array, const magma_int_t *incx_array,
            const double *beta, double** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count,
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

        for(int i=0; i<gsta.size()-1; i++){
           
           int idx = i%NSTREAMS;
           CUBLAS_CHECK(cublasSetStream(handle_cublas, stream[idx]));

           int ista = gsta[i];
           int nbatch = gsta[i+1]-ista;
           // convert from magma_int_t to int 
           int m = m_array[ista], n = n_array[ista];
           int lda = lda_array[ista], incx = incx_array[ista], incy = incy_array[ista]; 
           cublasOperation_t Trans = CUBLAS_OP_N ;
           if(trans=='T' || trans=='C'){
              Trans = CUBLAS_OP_T;
           }
           // https://docs.nvidia.com/cuda/cublas/index.html
           CUBLAS_CHECK(cublasDgemvBatched(handle_cublas,
                              Trans, 
                              m, n,
                              alpha,
                              &dev_A_array[ista], lda, // pointer should be on device
                              &dev_X_array[ista], incx,
                              beta,
                              &dev_Y_array[ista], incy,
                              nbatch));
        } // group

        // synchronize all streams
        int nstreams = (gsta.size()-1+NSTREAMS-1)/NSTREAMS; 
        for(int i=0; i<nstreams; i++){
           CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        }

        GPUmem.deallocate(dev_dtotal, total_dsize);
    }

    // complex 
    inline void xgemv_batch_gpu_stream(const char trans,
            const magma_int_t *m_array, const magma_int_t *n_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **A_array, const magma_int_t *lda_array, 
            const std::complex<double> **X_array, const magma_int_t *incx_array,
            const std::complex<double> *beta, std::complex<double>** Y_array, const magma_int_t *incy_array,
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemv_batch_gpu_stream"<<std::endl;
            return ;
        }
        std::cout << "NOT IMPLEMENTED YET!" << std::endl;
        exit(1);
    }

#endif

    // --- GEMM ---
    // double
    inline void xgemm_batch_gpu_magma(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const double *alpha, const double **a_array, const magma_int_t *lda_array, 
            const double **b_array, const magma_int_t *ldb_array,
            const double *beta, double **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_magma"<<std::endl;
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
        double** dev_a_array = (double**)dev_dtotal;
        double** dev_b_array = dev_a_array + batch_count;
        double** dev_c_array = dev_b_array + batch_count;

        GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_k, k_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));

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

        GPUmem.deallocate(dev_dtotal, total_dsize);
        GPUmem.deallocate(dev_itotal, total_isize);
    }

    // complex
    inline void xgemm_batch_gpu_magma(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **a_array, const magma_int_t *lda_array,
            const std::complex<double> **b_array, const magma_int_t *ldb_array,
            const std::complex<double> *beta, std::complex<double> **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_magma"<<std::endl;
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
        magmaDoubleComplex** dev_a_array = (magmaDoubleComplex**)dev_dtotal;
        magmaDoubleComplex** dev_b_array = dev_a_array + batch_count;
        magmaDoubleComplex** dev_c_array = dev_b_array + batch_count;

        GPUmem.to_gpu(dev_m, m_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_n, n_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_k, k_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_lda, lda_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_ldb, ldb_array, batch_count*sizeof(magma_int_t));
        GPUmem.to_gpu(dev_ldc, ldc_array, batch_count*sizeof(magma_int_t));
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

#ifndef HIP

    // --- GEMM GROUPED ---
    // double
    inline void xgemm_batch_gpu_grouped(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const double *alpha, const double **a_array, const magma_int_t *lda_array, 
            const double **b_array, const magma_int_t *ldb_array,
            const double *beta, double **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_grouped"<<std::endl;
            return ;
        }

        size_t total_dsize = 3*batch_count*sizeof(double*);
        void* dev_dtotal = GPUmem.allocate(total_dsize);
        double** dev_a_array = (double**)dev_dtotal;
        double** dev_b_array = dev_a_array + batch_count;
        double** dev_c_array = dev_b_array + batch_count;
        GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));
            
        for(int i=0; i<gsta.size()-1; i++){
           int ista = gsta[i];
           int nbatch = gsta[i+1]-ista;
           // convert from magma_int_t to int 
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
                              &dev_a_array[ista], lda, // pointer should be on device
                              &dev_b_array[ista], ldb,
                              beta,
                              &dev_c_array[ista], ldc,
                              nbatch));
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

        GPUmem.deallocate(dev_dtotal, total_dsize);
    }

    // complex
    inline void xgemm_batch_gpu_grouped(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **a_array, const magma_int_t *lda_array,
            const std::complex<double> **b_array, const magma_int_t *ldb_array,
            const std::complex<double> *beta, std::complex<double> **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_grouped"<<std::endl;
            return ;
        }
        std::cout << "NOT IMPLEMENTED YET!" << std::endl;
        exit(1);
    }

    // --- GEMM STREAM ---
    // double
    inline void xgemm_batch_gpu_stream(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const double *alpha, const double **a_array, const magma_int_t *lda_array, 
            const double **b_array, const magma_int_t *ldb_array,
            const double *beta, double **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_stream"<<std::endl;
            return ;
        }

        size_t total_dsize = 3*batch_count*sizeof(double*);
        void* dev_dtotal = GPUmem.allocate(total_dsize);
        double** dev_a_array = (double**)dev_dtotal;
        double** dev_b_array = dev_a_array + batch_count;
        double** dev_c_array = dev_b_array + batch_count;
        GPUmem.to_gpu(dev_a_array, a_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_b_array, b_array, batch_count*sizeof(double*));
        GPUmem.to_gpu(dev_c_array, c_array, batch_count*sizeof(double*));
            
        for(int i=0; i<gsta.size()-1; i++){

           int idx = i%NSTREAMS;
           CUBLAS_CHECK(cublasSetStream(handle_cublas, stream[idx])); 
               
           int ista = gsta[i];
           int nbatch = gsta[i+1]-ista;
           // convert from magma_int_t to int 
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
                              &dev_a_array[ista], lda, // pointer should be on device
                              &dev_b_array[ista], ldb,
                              beta,
                              &dev_c_array[ista], ldc,
                              nbatch));
        } // group

        // synchronize all streams
        int nstreams = (gsta.size()-1+NSTREAMS-1)/NSTREAMS; 
        for(int i=0; i<nstreams; i++){
           CUDA_CHECK(cudaStreamSynchronize(stream[i]));
        }

        GPUmem.deallocate(dev_dtotal, total_dsize);
    }

    // complex
    inline void xgemm_batch_gpu_stream(const char transa, const char transb, 
            const magma_int_t *m_array, const magma_int_t *n_array, const magma_int_t *k_array,
            const std::complex<double> *alpha, 
            const std::complex<double> **a_array, const magma_int_t *lda_array,
            const std::complex<double> **b_array, const magma_int_t *ldb_array,
            const std::complex<double> *beta, std::complex<double> **c_array, const magma_int_t *ldc_array, 
            const magma_int_t batch_count,
            const std::vector<int>& gsta){

        if(batch_count <= 0){
            std::cout<<"batch_count shoule > 0 in function xgemm_batch_gpu_stream"<<std::endl;
            return ;
        }
        std::cout << "NOT IMPLEMENTED YET!" << std::endl;
        exit(1);
    }

#endif

} // linalg

#endif

#endif//GPU
