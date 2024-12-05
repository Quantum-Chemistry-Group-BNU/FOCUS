#ifdef GPU

#include "gpu_env.h"
#include "../core/tools.h"

gpu_mem GPUmem = {};

#ifdef MAGMA
magma_queue_t magma_queue = 0;
#endif

cudaStream_t stream[NSTREAMS];
cublasHandle_t handle_cublas;

#ifdef NCCL
nccl_communicator nccl_comm;
#endif

void gpu_init(const int rank){
   if(rank == 0) std::cout << "\ngpu_init" << std::endl;

#ifdef MAGMA
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
   
   std::cout << "rank=" << rank << " num_gpus=" << num_gpus
      << " device_id=" << device_id << " magma_queue=" <<magma_queue
      << std::endl;
#endif

   CUDA_CHECK(cudaSetDevice(rank)); // important for nccl to work

   int cudaToolkitVersion;
   CUDA_CHECK(cudaRuntimeGetVersion(&cudaToolkitVersion));
   for(int i=0; i<NSTREAMS; i++){
      CUDA_CHECK(cudaStreamCreate(&stream[i]));
   }
   CUBLAS_CHECK(cublasCreate(&handle_cublas));
   int cublasVersion;
   CUBLAS_CHECK(cublasGetVersion(handle_cublas, &cublasVersion));
   if(rank == 0){
      std::cout << "CUDA Runtime version: " << cudaToolkitVersion << std::endl;
      std::cout << "CUBLAS version: " << cublasVersion << std::endl;
   }

#ifdef NCCL
   nccl_comm.init();
#endif
}

void gpu_finalize(){

   GPUmem.finalize();

#ifdef MAGMA
   magma_queue_destroy(magma_queue);
   magma_finalize();
#endif

   for(int i=0; i<NSTREAMS; i++){
      cudaStreamDestroy(stream[i]);
   }
   cublasDestroy(handle_cublas);

#ifdef NCCL
   nccl_comm.finalize();
#endif

}

#endif
