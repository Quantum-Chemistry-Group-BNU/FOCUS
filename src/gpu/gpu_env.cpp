#ifdef GPU

#include "gpu_env.h"
#include "../core/tools.h"

magma_queue_t magma_queue = 0;
gpu_mem GPUmem = {};

void gpu_init(const int rank){
   if(rank == 0) std::cout << "gpu_init" << std::endl;
   magma_queue = 0;
   magma_device_t device_id = -1;

   magma_device_t devices[MAX_GPUS];
   magma_int_t num_gpus = 0;

   magma_init();
   magma_getdevices(devices, MAX_GPUS, &num_gpus);
   if(num_gpus == 0){
      std::cout<<"error: no GPU available!"<<std::endl;
      exit(1);
   }
   magma_setdevice(rank % num_gpus);
   magma_getdevice(&device_id);

   magma_queue_create(device_id, &magma_queue);

   /*
   void* addr;
   size_t size = 1024;
   CUDA_CHECK(cudaMalloc((void**)&addr, size));
   CUDA_CHECK(cudaFree(addr));
   std::cout << "run succesfully" << std::endl;
   */

   std::cout << "rank=" << rank << " num_gpus=" << num_gpus
      << " device_id=" << device_id << " magma_queue=" <<magma_queue
      << std::endl;
}

void gpu_finalize(){
   magma_queue_destroy(magma_queue);
   magma_finalize();
}

#endif
