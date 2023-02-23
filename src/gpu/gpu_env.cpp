#ifdef GPU

#include "gpu_env.h"
#include "../core/tools.h"

extern magma_queue_t magma_queue =0;
extern void* dev_addr = nullptr;
extern size_t gpumem_tot = 0;

void gpu_init(int rank)
{
   magma_queue = 0;
   int device_id = -1;

   magma_device_t devices[MAXGPUS];
   magma_int_t num_gpus=0;

   magma_init();
   magma_getdevices(devices, MAXGPUS, &num_gpus);
   if(num_gpus == 0)
   {
       std::cout<<"no GPU avail !"<<std::endl;
       exit(1);
   }
   magma_setdevice(rank % num_gpus);
   magma_getdevice(&device_id);

   magma_queue_create(device_id, &magma_queue);
   std::cout<<"rank: "<< rank <<"; num_gpus="<<num_gpus<<"; device_id: "<< device_id <<"; magma_queue="<<magma_queue << std::endl;

   // allocate GPU memory
   size_t free, total;
#ifdef USE_HIP
	 HIP_CHECK(hipMemGetInfo(&free, &total));
#else
   CUDA_CHECK(cudaMemGetInfo( &free, &total ));
#endif
   if(free <= MAX_GPU_PAGE){
      std::cout << "error: GPU memory is too small!" << std::endl;
      exit(1);
   }
   gpumem_tot = free - MAX_GPU_PAGE;
   std::cout << "allocated gpumem_tot (GB) = " 
             << tools::sizeGB<std::byte>(gpumem_tot)
             << std::endl;

#ifdef USE_HIP
	 HIP_CHECK(hipMalloc((void**)&dev_addr, gpumem_tot));
#else
#if defined(USE_CUDA_OPERATION)
				CUDA_CHECK(cudaMalloc((void**)&dev_addr, gpumem_tot));
#else//MAGMA
				MAGMA_CHECK(magma_malloc((void**)&dev_addr, gpumem_tot));
#endif// USE_CUDA_OPERATION
#endif//USE_HIP
}

void gpu_clean()
{
#ifdef USE_HIP
   HIP_CHECK(hipFree(dev_addr));
#else
#if defined(USE_CUDA_OPERATION)
	 CUDA_CHECK(cudaFree(dev_addr));
#else//MAGMA
	 MAGMA_CHECK(magma_free(dev_addr));
#endif// USE_CUDA_OPERATION
#endif//USE_HIP
	 dev_addr = nullptr;
   magma_queue_destroy(magma_queue);
   magma_finalize();
}

#endif
