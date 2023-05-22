#ifdef GPU

#ifndef GPU_ENV_H
#define GPU_ENV_H

#include <magma_v2.h>
#include "gpu_mem.h"
#include "gpu_nccl.h"

const int MAX_GPUS = 128;
extern const int MAX_GPUS;

#ifndef HIP
extern cublasHandle_t handle_cublas;
#endif

extern magma_queue_t magma_queue;
extern gpu_mem GPUmem; 

#ifdef NCCL
extern nccl_communicator nccl_comm;
#endif

void gpu_init(const int rank);
void gpu_finalize();

#endif //GPU_ENV_H

#endif //GPU
