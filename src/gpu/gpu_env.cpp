#ifdef GPU

#include "gpu_env.h"

extern int nqueue=NQUEUE;
extern magma_queue_t magma_queue =0;
extern magma_queue_t magma_queue_array[NQUEUE]={0};

#endif
