#ifdef GPU

#ifndef GPU_ENV_H
#define GPU_ENV_H

#include "gpu_mem.h"
#include "gpu_nccl.h"

extern gpu_mem GPUmem; 

#ifdef MAGMA
#include <magma_v2.h>
const int MAX_GPUS = 16; // MAX GPU PER NODE
extern const int MAX_GPUS;
extern magma_queue_t magma_queue;
#endif

// global variables for CUBLAS
#define NSTREAMS 100
extern cudaStream_t stream[NSTREAMS];
extern cublasHandle_t handle_cublas;

#ifdef NCCL
extern nccl_communicator nccl_comm;
#endif

void gpu_init(const int rank);
void gpu_finalize();

#endif //GPU_ENV_H

#endif //GPU
