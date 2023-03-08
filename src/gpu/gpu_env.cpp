#ifdef GPU

#include "gpu_env.h"
#include "../core/tools.h"

magma_queue_t magma_queue = 0;
gpu_mem GPUmem = {};

void gpu_init(int rank)
{
   magma_queue = 0;
   magma_device_t device_id = -1;

   magma_device_t devices[MAXGPUS];
   magma_int_t num_gpus = 0;

   magma_init();
   magma_getdevices(devices, MAXGPUS, &num_gpus);
   if(num_gpus == 0)
   {
      std::cout<<"error: no GPU available!"<<std::endl;
      exit(1);
   }
   magma_setdevice(rank % num_gpus);
   magma_getdevice(&device_id);

   magma_queue_create(device_id, &magma_queue);
   std::cout << "rank =" << rank << " num_gpus=" << num_gpus
             << " device_id=" << device_id << " magma_queue=" <<magma_queue 
             << std::endl;

   GPUmem.init();
}

void gpu_clean()
{
   GPUmem.free();
   magma_queue_destroy(magma_queue);
   magma_finalize();
}

#endif
