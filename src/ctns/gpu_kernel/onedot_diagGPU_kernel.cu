#include "onedot_diagGPU_kernel.h"
#include <iostream>
#include <cuda_runtime.h>

std::pair<dim3,dim3> get_partition1(const size_t ndim){
   const size_t nthreads = 512;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((ndim+nthreads-1)/nthreads);
   return std::make_pair(dimBlock,dimGrid);
}

__device__ void get_position1(const size_t idx, 
      const size_t nblk, 
      const size_t* dev_dims,
      int& i,
      size_t* dims,
      int* id){
   // determine the block
   i = nblk-1;
   for(int j=0; j<nblk; j++){
      if(idx < dev_dims[4*j+3]){
         i = j-1;
         break;
      } 
   }
   dims[0] = dev_dims[4*i];   // r 
   dims[1] = dev_dims[4*i+1]; // c
   dims[2] = dev_dims[4*i+2]; // m 
   // dtermine im,ic,ir
   size_t idx3 = idx - dev_dims[4*i+3];
   id[0] = idx3 % dims[0];
   size_t idx2 = idx3 / dims[0];
   id[1] = idx2 % dims[1];
   id[2] = idx2 / dims[1];
}

// H[local]
__global__ void diagGPU_local_kernel1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_copaddr){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t l = dev_dims[4*nblk+3*i]   + id[0]*(dims[0]+1);
      size_t r = dev_dims[4*nblk+3*i+1] + id[1]*(dims[1]+1);
      size_t c = dev_dims[4*nblk+3*i+2] + id[2]*(dims[2]+1);
      dev_diag[idx] += dev_lopaddr[l]
         + dev_ropaddr[r]
         + dev_copaddr[c];
   }
}
__global__ void diagGPU_local_kernel1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_copaddr){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t l = dev_dims[4*nblk+3*i]   + id[0]*(dims[0]+1);
      size_t r = dev_dims[4*nblk+3*i+1] + id[1]*(dims[1]+1);
      size_t c = dev_dims[4*nblk+3*i+2] + id[2]*(dims[2]+1);
      dev_diag[idx] += dev_lopaddr[l].x
         + dev_ropaddr[r].x
         + dev_copaddr[c].x;
   }
}
template <>
void ctns::onedot_diagGPU_local(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_copaddr){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel1<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_copaddr);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
template <>
void ctns::onedot_diagGPU_local(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_copaddr){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel1<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_copaddr);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}

// O1O2
__global__ void diagGPU_O1O2_kernel1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt,
      const int i1,
      const int i2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[4*nblk+2*i]   + id[i1]*(dims[i1]+1);
      size_t o2 = dev_dims[4*nblk+2*i+1] + id[i2]*(dims[i2]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_O1O2_kernel1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt,
      const int i1,
      const int i2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[4*nblk+2*i]   + id[i1]*(dims[i1]+1);
      size_t o2 = dev_dims[4*nblk+2*i+1] + id[i2]*(dims[i2]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::onedot_diagGPU_O1O2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt,
      const int i1,
      const int i2){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_O1O2_kernel1<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt, i1, i2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
template <>
void ctns::onedot_diagGPU_O1O2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt,
      const int i1,
      const int i2){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_O1O2_kernel1<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt, i1, i2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}

// O1O2_su2
__global__ void diagGPU_O1O2_kernel1_su2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double* dev_fac,
      const int i1,
      const int i2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[4*nblk+2*i]   + id[i1]*(dims[i1]+1);
      size_t o2 = dev_dims[4*nblk+2*i+1] + id[i2]*(dims[i2]+1);
      if(std::abs(dev_fac[i]) < ctns::thresh_diag_angular2) return;
      dev_diag[idx] += dev_fac[i]*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_O1O2_kernel1_su2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double* dev_fac,
      const int i1,
      const int i2){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[3];
      int id[3];
      get_position1(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[4*nblk+2*i]   + id[i1]*(dims[i1]+1);
      size_t o2 = dev_dims[4*nblk+2*i+1] + id[i2]*(dims[i2]+1);
      if(std::abs(dev_fac[i]) < ctns::thresh_diag_angular2) return;
      dev_diag[idx] += dev_fac[i]*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::onedot_diagGPU_O1O2_su2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double* dev_fac,
      const int i1,
      const int i2){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_O1O2_kernel1_su2<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, dev_fac, i1, i2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
template <>
void ctns::onedot_diagGPU_O1O2_su2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double* dev_fac,
      const int i1,
      const int i2){
   auto part = get_partition1(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_O1O2_kernel1_su2<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, dev_fac, i1, i2);
   cudaError_t cudaError = cudaGetLastError();
   if (cudaError != cudaSuccess) {
      printf("CUDA error: %s\n", cudaGetErrorString(cudaError));
      exit(1);
   }
}
