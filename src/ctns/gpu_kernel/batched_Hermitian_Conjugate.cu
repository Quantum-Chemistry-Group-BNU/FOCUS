#include "batched_Hermitian_Conjugate.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void  batched_Hermitian_Conjugate_kernel(const size_t nblks,
      const size_t* dev_offs,
      const int* dev_dims,
      const double* dev_facs,
      const double* dev_qops1,
      double* dev_qops2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < nblks){
      size_t off2 = dev_offs[2*idx];
      size_t off1 = dev_offs[2*idx+1];
      size_t rows = dev_dims[2*idx];
      size_t cols = dev_dims[2*idx+1];
      double fac = dev_facs[idx];
      for(int j=0; j<cols; j++){
         for(int i=0; i<rows; i++){
            dev_qops2[off2+j*rows+i] = fac*dev_qops1[off1+i*cols+j];
         }
      }
   } // idx 
}
__global__ void  batched_Hermitian_Conjugate_kernel(const size_t nblks,
      const size_t* dev_offs,
      const int* dev_dims,
      const COMPLX* dev_facs,
      const COMPLX* dev_qops1,
      COMPLX* dev_qops2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < nblks){
      size_t off2 = dev_offs[2*idx];
      size_t off1 = dev_offs[2*idx+1];
      size_t rows = dev_dims[2*idx];
      size_t cols = dev_dims[2*idx+1];
      COMPLX fac = dev_facs[idx];
      for(int j=0; j<cols; j++){
         for(int i=0; i<rows; i++){
            dev_qops2[off2+j*rows+i] = cuCmul(fac,cuConj(dev_qops1[off1+i*cols+j]));
         }
      }
   } // idx 
}

template <>
void ctns::batched_Hermitian_Conjugate(const size_t nblks,
      const size_t* dev_offs,
      const int* dev_dims,
      const double* dev_facs,
      const double* dev_qops1,
      double* dev_qops2){
   const size_t nthreads = 512;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((nblks+nthreads-1)/nthreads);
   batched_Hermitian_Conjugate_kernel<<<dimGrid,dimBlock>>>(nblks, dev_offs,
         dev_dims, dev_facs, dev_qops1, dev_qops2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
template <>
void ctns::batched_Hermitian_Conjugate(const size_t nblks,
      const size_t* dev_offs,
      const int* dev_dims,
      const COMPLX* dev_facs,
      const COMPLX* dev_qops1,
      COMPLX* dev_qops2){
   const size_t nthreads = 512;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((nblks+nthreads-1)/nthreads);
   batched_Hermitian_Conjugate_kernel<<<dimGrid,dimBlock>>>(nblks, dev_offs,
         dev_dims, dev_facs, dev_qops1, dev_qops2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
