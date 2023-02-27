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

#endif //USE_HIP

#endif //GPU_CHECK_H

#endif //GPU
