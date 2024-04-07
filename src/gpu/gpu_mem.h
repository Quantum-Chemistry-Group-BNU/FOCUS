#ifdef GPU

#ifndef GPU_MEM_H
#define GPU_MEM_H

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#else
#include <cuda_runtime.h>
#include <cuda.h>
#endif //USE_HIP

#include <iostream>
#include "gpu_check.h"

const size_t MAX_GPU_PAGE = 128*1024*1024;
extern const size_t MAX_GPU_PAGE;

// interface
class gpu_mem{
   public:
      gpu_mem(): _used(0) {};

      void finalize(){
         _used = 0;
      }

      size_t available(const int rank){
         size_t avail, total;
#ifdef USE_HIP
         HIP_CHECK(hipMemGetInfo(&avail, &total));
#else
         CUDA_CHECK(cudaMemGetInfo(&avail, &total));
#endif
         if(avail <= MAX_GPU_PAGE){
            std::cout << "error: GPU memory is too small on rank=" << rank << std::endl;
            exit(1);
         }
         avail -= MAX_GPU_PAGE;
         return avail; 
      }

      void* allocate(const size_t size){
         void *addr;
#ifdef USE_HIP
         HIP_CHECK(hipMalloc((void**)&addr, size));
#else
         CUDA_CHECK(cudaMalloc((void**)&addr, size));
#endif //USE_HIP
         _used += size; 
         return addr;
      }

      void deallocate(void *ptr, const size_t size){
         if(ptr == nullptr) return;
         if(_used < size){
            std::cout << "error in deallocate: _used=" << _used << " size=" << size << std::endl;
            exit(1);
         }
         _used -= size;
#ifdef USE_HIP
         HIP_CHECK(hipFree(ptr));
#else
         CUDA_CHECK(cudaFree(ptr));
#endif //USE_HIP
      }

      void memset(void *ptr, const size_t size){
#ifdef USE_HIP
         HIP_CHECK(hipMemset(ptr, 0, size));
#else
         CUDA_CHECK(cudaMemset(ptr, 0, size));
#endif //USE_HIP
      }

      void to_gpu(void *dev_ptr, const void *ptr, const size_t size){
#ifdef USE_HIP
         HIP_CHECK(hipMemcpy(dev_ptr, ptr, size, hipMemcpyHostToDevice));
#else
         CUDA_CHECK(cudaMemcpy(dev_ptr, ptr, size, cudaMemcpyHostToDevice));
#endif //USE_HIP
      }

      void to_cpu(void *ptr, const void *dev_ptr, const size_t size){
#ifdef USE_HIP
         HIP_CHECK(hipMemcpy(ptr, dev_ptr, size, hipMemcpyDeviceToHost));
#else
         CUDA_CHECK(cudaMemcpy(ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
#endif //USE_HIP
      }

      void sync(){
#ifdef USE_HIP
         hipDeviceSynchronize();
#else
         cudaDeviceSynchronize();
#endif
      }

      size_t used() const{ return _used; }
   private:
      size_t _used = 0;
};

#endif // GPU_MEM_H

#endif // GPU
