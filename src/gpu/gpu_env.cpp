#ifdef GPU

#include "gpu_env.h"
#include "../core/tools.h"

magma_queue_t magma_queue = 0;
gpu_mem GPUmem = {};

#ifndef USE_HIP
cudaStream_t stream[NSTREAMS];
cublasHandle_t handle_cublas;
#endif

#ifdef NCCL
nccl_communicator nccl_comm;
#endif

void gpu_init(const int rank){
   if(rank == 0) std::cout << "\ngpu_init" << std::endl;
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

#ifndef USE_HIP
   int cudaToolkitVersion;
   cudaRuntimeGetVersion(&cudaToolkitVersion);
   for(int i=0; i<NSTREAMS; i++){
      cudaStreamCreate(&stream[i]);
   }
   cublasCreate(&handle_cublas);
   int cublasVersion;
   cublasGetVersion(handle_cublas, &cublasVersion);
   if(rank == 0){
      std::cout << "CUDA Runtime version: " << cudaToolkitVersion << std::endl;
      std::cout << "CUBLAS version: " << cublasVersion << std::endl;
   }
#endif

#ifdef NCCL
   nccl_comm.init();
#endif

   std::cout << "rank=" << rank << " num_gpus=" << num_gpus
      << " device_id=" << device_id << " magma_queue=" <<magma_queue
      << std::endl;
}

void gpu_finalize(){

#ifndef USE_HIP
   for(int i=0; i<NSTREAMS; i++){
      cudaStreamDestroy(stream[i]);
   }
   cublasDestroy(handle_cublas);
#endif

#ifdef NCCL
   nccl_comm.finalize();
#endif

   GPUmem.finalize();

   magma_queue_destroy(magma_queue);
   magma_finalize();
}

#endif
