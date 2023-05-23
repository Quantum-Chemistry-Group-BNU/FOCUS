#ifdef GPU

#ifndef GPU_CHECK_H
#define GPU_CHECK_H

#define MAGMA_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
#err, __FILE__, __LINE__,                               \
                    (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

#ifdef USE_HIP
#define HIP_CHECK(err)                    \
    if(err != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(err),         \
                err,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#else
#define CUDA_CHECK( err )                                                 \
    do {                                                                     \
        cudaError_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
#err, __FILE__, __LINE__,                               \
                    (long long) err_, cudaGetErrorString(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

//cuBLAS API errors
static const char *_cublasGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "<unknown error in cublas>";
}

//Macro Cublas error check, check error message during a Cublas launch or cuda api call
#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        if(err!=CUBLAS_STATUS_SUCCESS) {                                       \
            printf("Cuda failure %i:%s:%s:%d \n",err,_cublasGetErrorEnum(err), \
                    __FILE__,__LINE__);                                        \
            exit(-1);                                                          \
        }                                                                      \
    } while( 0 )

#endif //USE_HIP

#endif //GPU_CHECK_H

#endif //GPU
