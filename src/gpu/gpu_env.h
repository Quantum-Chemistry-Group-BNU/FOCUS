#ifdef GPU
#ifndef GPU_ENV_H
#define GPU_ENV_H

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <magma_v2.h>
#else

#include <cuda_runtime.h>
#include <cuda.h>
#include <magma_v2.h>
#endif //USE_HIP

extern magma_queue_t magma_queue;

extern void* dev_addr;

extern size_t gpumem_tot;

const int MAXGPUS = 16;
extern const int MAXGPUS;

const size_t MAX_GPU_PAGE= 128*1024*1024;
extern const size_t MAX_GPU_PAGE;

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
#endif //GPU_ENV_H
#endif //GPU

