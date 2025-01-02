#ifdef GPU

#ifndef GPU_MEM_H
#define GPU_MEM_H

#include <cuda_runtime.h>
#include <cuda.h>
#include <cassert>
#include <iostream>
#include "gpu_check.h"

const size_t MAX_GPU_PAGE = 256*1024*1024;
extern const size_t MAX_GPU_PAGE;

// interface
class gpu_mem{
   public:
      gpu_mem(): _used(0) {};

      void finalize(){
         assert(_used == 0);
         _used = 0;
      }

      size_t available(const int rank){
         _rank = rank;
         size_t avail, total;
         CUDA_CHECK(cudaMemGetInfo(&avail, &total));
         if(avail <= MAX_GPU_PAGE){
            std::cout << "error: GPU memory is too small on rank=" << rank << std::endl;
            exit(1);
         }
         avail -= MAX_GPU_PAGE;
         return avail; 
      }

      void* allocate(const size_t size){
         //-----------------------------
         // ZL@2025/01/02: check memory 
         //-----------------------------
         size_t avail, total;
         CUDA_CHECK(cudaMemGetInfo(&avail, &total));
         if(size > avail){
            std::cout << "error: no enough memory on GPU!:"
               << " rank=" << _rank 
               << " total=" << total/1024.0/1024.0/1024.0
               << " avail=" << avail/1024.0/1024.0/1024.0
               << " used=" << _used/1024.0/1024.0/1024.0
               << " size[need]=" << size/1024.0/1024.0/1024.0 
               << std::endl;
         }
         //-----------------------------
         void *addr;
         CUDA_CHECK(cudaMalloc((void**)&addr, size));
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
         CUDA_CHECK(cudaFree(ptr));
      }

      void memset(void *ptr, const size_t size){
         CUDA_CHECK(cudaMemset(ptr, 0, size));
      }

      void to_gpu(void *dev_ptr, const void *ptr, const size_t size){
         CUDA_CHECK(cudaMemcpy(dev_ptr, ptr, size, cudaMemcpyHostToDevice));
      }

      void to_cpu(void *ptr, const void *dev_ptr, const size_t size){
         CUDA_CHECK(cudaMemcpy(ptr, dev_ptr, size, cudaMemcpyDeviceToHost));
      }

      void sync(){
         cudaDeviceSynchronize();
      }

      size_t used() const{ return _used; }
   private:
      size_t _used = 0;
      int _rank = -1;
};

#endif // GPU_MEM_H

#endif // GPU
