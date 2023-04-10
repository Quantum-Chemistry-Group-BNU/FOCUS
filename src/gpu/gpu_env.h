#ifdef GPU

#ifndef GPU_ENV_H
#define GPU_ENV_H

#include "gpu_mem.h"
#include <magma_v2.h>

extern magma_queue_t magma_queue;
extern gpu_mem GPUmem; 

const int MAXGPUS = 100;
extern const int MAXGPUS;

void gpu_init(const int rank);
void gpu_finalize();

#endif //GPU_ENV_H

#endif //GPU
