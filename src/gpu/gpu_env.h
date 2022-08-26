#ifdef GPU

#ifndef GPU_ENV_H
#define GPU_ENV_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <magma_v2.h>

#define NQUEUE 2
extern int nqueue;
extern magma_queue_t magma_queue;
extern magma_queue_t magma_queue_array[NQUEUE];

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

#endif

#endif
