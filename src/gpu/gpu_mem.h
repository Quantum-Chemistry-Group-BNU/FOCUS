#ifdef GPU

#ifndef GPU_MEM_H
#define GPU_MEM_H

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <magma_v2.h>
#else
#include <cuda_runtime.h>
#include <cuda.h>
#include <magma_v2.h>
#endif //USE_HIP

#include "../core/tools.h" 
#include "gpu_check.h"

const size_t MAX_GPU_PAGE= 128*1024*1024;
extern const size_t MAX_GPU_PAGE;

class gpu_mem{
   public:
      gpu_mem(): _size(0), _used(0), _addr(nullptr) {}

      void init(){
         // allocate GPU memory
         size_t avail, total;
#ifdef USE_HIP
         HIP_CHECK(hipMemGetInfo(&avail, &total));
#else
         CUDA_CHECK(cudaMemGetInfo(&avail, &total));
#endif
         if(avail <= MAX_GPU_PAGE){
            std::cout << "error: GPU memory is too small!" << std::endl;
            exit(1);
         }
         _size = avail - MAX_GPU_PAGE;
         std::cout << "allocated gpu mem size (GB) = "
            << tools::sizeGB<std::byte>(_size)
            << std::endl;

#ifdef USE_HIP
         HIP_CHECK(hipMalloc((void**)&_addr, _size));
#else
         CUDA_CHECK(cudaMalloc((void**)&_addr, _size));
#endif //USE_HIP

      }

      void free(){
#ifdef USE_HIP
         HIP_CHECK(hipFree(_addr));
#else
         CUDA_CHECK(cudaFree(_addr));
#endif //USE_HIP
         _addr = nullptr;
         _size = 0;
         _used = 0;
      }

      void* allocate(size_t n){
         if(_used + n >= _size){
            std::cout << "error: exceeding allowed GPU memory" << std::endl;
            exit(1);
         }else{
            _used = _used + n;
            return _addr + _used - n;
         }
      }

      void deallocate(void *ptr, size_t n){
         if(n == 0) return;
         if(_used < n || ptr != _addr + _used - n){
            std::cout << "error: deallocation not happening in reverse order"
                      << " (_used < n)=" << (_used < n) 
                      << " (ptr != _addr + _used - n)" << (ptr != _addr + _used - n)
                      << std::endl;
            exit(1);
         }else{
            _used = _used - n;
         }
      }

      size_t size() const{ return _size; }
      size_t used() const{ return _used; }
   private:
      size_t _size = 0;
      size_t _used = 0;
      void* _addr = nullptr;
};

#endif // GPU_MEM_H

#endif // GPU
