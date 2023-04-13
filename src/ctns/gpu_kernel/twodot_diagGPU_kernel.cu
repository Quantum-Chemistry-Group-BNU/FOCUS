#include "twodot_diagGPU_kernel.h"
#include <iostream>

std::pair<dim3,dim3> get_partition(const size_t nblk){
   const int nthreads = 512;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((nblk+nthreads-1)/nthreads);
   return std::make_pair(dimBlock,dimGrid);
}

// H[local]
__global__ void diagGPU_local_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_c1opaddr,
      const double* dev_c2opaddr){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t loff = dev_dims[5*nblk+4*i];
      size_t roff = dev_dims[5*nblk+4*i+1];
      size_t c1off = dev_dims[5*nblk+4*i+2];
      size_t c2off = dev_dims[5*nblk+4*i+3];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += dev_lopaddr[loff+ir*(rdim+1)]
                     + dev_ropaddr[roff+ic*(cdim+1)]
                     + dev_c1opaddr[c1off+im*(mdim+1)]
                     + dev_c2opaddr[c2off+iv*(vdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_local_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_c1opaddr,
      const COMPLX* dev_c2opaddr){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t loff = dev_dims[5*nblk+4*i];
      size_t roff = dev_dims[5*nblk+4*i+1];
      size_t c1off = dev_dims[5*nblk+4*i+2];
      size_t c2off = dev_dims[5*nblk+4*i+3];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += dev_lopaddr[loff+ir*(rdim+1)].x
                     + dev_ropaddr[roff+ic*(cdim+1)].x
                     + dev_c1opaddr[c1off+im*(mdim+1)].x
                     + dev_c2opaddr[c2off+iv*(vdim+1)].x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_local(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_lopaddr,
      const double* dev_ropaddr,
      const double* dev_c1opaddr,
      const double* dev_c2opaddr){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
}
template <>
void ctns::twodot_diagGPU_local(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_lopaddr,
      const COMPLX* dev_ropaddr,
      const COMPLX* dev_c1opaddr,
      const COMPLX* dev_c2opaddr){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
}

// Ol*Oc1
__global__ void diagGPU_OlOc1_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+ir*(rdim+1)]*dev_opaddr2[off2+im*(mdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_OlOc1_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+ir*(rdim+1)],dev_opaddr2[off2+im*(mdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_OlOc1(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc1_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOc1(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc1_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Oc2
__global__ void diagGPU_OlOc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+ir*(rdim+1)]*dev_opaddr2[off2+iv*(vdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_OlOc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+ir*(rdim+1)],dev_opaddr2[off2+iv*(vdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_OlOc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Or
__global__ void diagGPU_OlOr_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+ir*(rdim+1)]*dev_opaddr2[off2+ic*(cdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_OlOr_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+ir*(rdim+1)],dev_opaddr2[off2+ic*(cdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_OlOr(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOr_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_OlOr(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOr_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Oc2
__global__ void diagGPU_Oc1Oc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+im*(mdim+1)]*dev_opaddr2[off2+iv*(vdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_Oc1Oc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+im*(mdim+1)],dev_opaddr2[off2+iv*(vdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_Oc1Oc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Oc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc1Oc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Oc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Or
__global__ void diagGPU_Oc1Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+im*(mdim+1)]*dev_opaddr2[off2+ic*(cdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_Oc1Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+im*(mdim+1)],dev_opaddr2[off2+ic*(cdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_Oc1Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Or_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc1Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Or_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc2*Or
__global__ void diagGPU_Oc2Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*dev_opaddr1[off1+iv*(vdim+1)]*dev_opaddr2[off2+ic*(cdim+1)];
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
__global__ void diagGPU_Oc2Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      size_t off1 = dev_dims[5*nblk+2*i];
      size_t off2 = dev_dims[5*nblk+2*i+1];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*COMPLX_MUL(dev_opaddr1[off1+iv*(vdim+1)],dev_opaddr2[off2+ic*(cdim+1)]).x;
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <>
void ctns::twodot_diagGPU_Oc2Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc2Or_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}
template <>
void ctns::twodot_diagGPU_Oc2Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc2Or_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

/*
// test
int main(){
const size_t nblk = 1;
double* dev_diag = nullptr;
size_t* dev_dims = nullptr;
{
using Tm = double;
Tm* dev_lopaddr = nullptr;
Tm* dev_ropaddr = nullptr;
Tm* dev_c1opaddr = nullptr;
Tm* dev_c2opaddr = nullptr;
ctns::twodot_diagGPU_local(nblk, dev_diag, dev_dims, dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
ctns::twodot_diagGPU_OlOc1(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_OlOc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_OlOr(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc1Oc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc1Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc2Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
}
{
using Tm = std::complex<double>;
Tm* dev_lopaddr = nullptr;
Tm* dev_ropaddr = nullptr;
Tm* dev_c1opaddr = nullptr;
Tm* dev_c2opaddr = nullptr;
ctns::twodot_diagGPU_local(nblk, dev_diag, dev_dims, dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
ctns::twodot_diagGPU_OlOc1(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_OlOc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_OlOr(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc1Oc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc1Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
ctns::twodot_diagGPU_Oc2Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr, 1.0);
}
return 0;
}
 */
