#ifdef GPU

#ifndef GPU_ENV_H
#define GPU_ENV_H

#include "gpu_mem.h"

extern magma_queue_t magma_queue;
extern gpu_mem gpumem; 

const int MAXGPUS = 16;
extern const int MAXGPUS;

void gpu_init(int rank);
void gpu_clean();

#endif //GPU_ENV_H

#endif //GPU
