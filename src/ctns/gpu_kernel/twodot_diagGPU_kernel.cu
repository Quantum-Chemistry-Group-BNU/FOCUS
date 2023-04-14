#include "twodot_diagGPU_kernel.h"
#include <iostream>

std::pair<dim3,dim3> get_partition(const size_t ndim){
   const size_t nthreads = 1024;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((ndim+nthreads-1)/nthreads);
   return std::make_pair(dimBlock,dimGrid);
}

__device__ void get_position(const size_t idx, 
      const size_t nblk, 
      const size_t* dev_dims,
      int& i,
      size_t* dims,
      int* id){
   // determine the block
   i = nblk-1;
   for(int j=0; j<nblk; j++){
      if(idx < dev_dims[5*j+4]){
         i = j-1;
         break;
      } 
   }
   dims[0] = dev_dims[5*i];   // r 
   dims[1] = dev_dims[5*i+1]; // c
   dims[2] = dev_dims[5*i+2]; // m 
   dims[3] = dev_dims[5*i+3]; // v
   // dtermine iv,im,ic,ir
   size_t idx4 = idx - dev_dims[5*i+4];
   id[0] = idx4 % dims[0];
   size_t idx3 = idx4 / dims[0];
   id[1] = idx3 % dims[1];
   size_t idx2 = idx3 / dims[1];
   id[2] = idx2 % dims[2];
   id[3] = idx2 / dims[2];
}

// H[local]
__global__ void diagGPU_local_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_c1opaddr,
      const double* dev_c2opaddr){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t l  = dev_dims[5*nblk+4*i]   + id[0]*(dims[0]+1);
      size_t r  = dev_dims[5*nblk+4*i+1] + id[1]*(dims[1]+1);
      size_t c1 = dev_dims[5*nblk+4*i+2] + id[2]*(dims[2]+1);
      size_t c2 = dev_dims[5*nblk+4*i+3] + id[3]*(dims[3]+1);
      dev_diag[idx] += dev_lopaddr[l]
                     + dev_ropaddr[r]
                     + dev_c1opaddr[c1]
                     + dev_c2opaddr[c2];
   }
}
__global__ void diagGPU_local_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_c1opaddr,
      const COMPLX* dev_c2opaddr){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t l  = dev_dims[5*nblk+4*i]   + id[0]*(dims[0]+1);
      size_t r  = dev_dims[5*nblk+4*i+1] + id[1]*(dims[1]+1);
      size_t c1 = dev_dims[5*nblk+4*i+2] + id[2]*(dims[2]+1);
      size_t c2 = dev_dims[5*nblk+4*i+3] + id[3]*(dims[3]+1);
      dev_diag[idx] += dev_lopaddr[l].x
                     + dev_ropaddr[r].x
                     + dev_c1opaddr[c1].x
                     + dev_c2opaddr[c2].x;
   }
}
template <>
void ctns::twodot_diagGPU_local(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_c1opaddr,
      const double* dev_c2opaddr){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
}
template <>
void ctns::twodot_diagGPU_local(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_c1opaddr,
      const COMPLX* dev_c2opaddr){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
}

// Ol*Oc1
__global__ void diagGPU_OlOc1_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[2]*(dims[2]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_OlOc1_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[2]*(dims[2]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_OlOc1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc1_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOc1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc1_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Oc2
__global__ void diagGPU_OlOc2_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[3]*(dims[3]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_OlOc2_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[3]*(dims[3]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_OlOc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc2_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc2_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Or
__global__ void diagGPU_OlOr_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_OlOr_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[0]*(dims[0]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_OlOr(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOr_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOr(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOr_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Oc2
__global__ void diagGPU_Oc1Oc2_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[2]*(dims[2]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[3]*(dims[3]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_Oc1Oc2_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[2]*(dims[2]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[3]*(dims[3]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_Oc1Oc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Oc2_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc1Oc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Oc2_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Or
__global__ void diagGPU_Oc1Or_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[2]*(dims[2]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_Oc1Or_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[2]*(dims[2]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_Oc1Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Or_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc1Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Or_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc2*Or
__global__ void diagGPU_Oc2Or_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[3]*(dims[3]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*dev_opaddr1[o1]*dev_opaddr2[o2];
   }
}
__global__ void diagGPU_Oc2Or_kernel(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(idx < ndim){
      int i;
      size_t dims[4];
      int id[4];
      get_position(idx, nblk, dev_dims, i, dims, id);
      size_t o1 = dev_dims[5*nblk+2*i]   + id[3]*(dims[3]+1);
      size_t o2 = dev_dims[5*nblk+2*i+1] + id[1]*(dims[1]+1);
      dev_diag[idx] += wt*COMPLX_MUL(dev_opaddr1[o1],dev_opaddr2[o2]).x;
   }
}
template <>
void ctns::twodot_diagGPU_Oc2Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc2Or_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc2Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(ndim);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc2Or_kernel<<<dimGrid,dimBlock>>>(nblk, ndim, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
