#include <complex>
#include "twodot_diagGPU_kernel.h"

template <typename Tm>
__device__ double real_part(Tm x){}
__device__ inline double real_part(const double& x){ return x; }
__device__ inline double real_part(std::complex<double>& x){ 
   const double* xArg = reinterpret_cast<const double*>(& x);
   return xArg[0];
}
   
std::pair<dim3,dim3> get_partition(const size_t nblk){
   const int nthreads = 512;
   dim3 dimBlock(nthreads);
   dim3 dimGrid((nblk+nthreads-1)/nthreads);
   return std::make_pair(dimBlock,dimGrid);
}
 
// H[local]
template <typename Tm>
__global__ void diagGPU_local_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_lopaddr,
      const Tm* dev_ropaddr,
      const Tm* dev_c1opaddr,
      const Tm* dev_c2opaddr){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += real_part(dev_lopaddr[ir*(rdim+1)])
                     + real_part(dev_ropaddr[ic*(cdim+1)])
                     + real_part(dev_c1opaddr[im*(mdim+1)])
                     + real_part(dev_c2opaddr[iv*(vdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void ctns::twodot_diagGPU_local(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_lopaddr,
      const Tm* dev_ropaddr,
      const Tm* dev_c1opaddr,
      const Tm* dev_c2opaddr){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_local_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
}

// Ol*Oc1
template <typename Tm>
__global__ void diagGPU_OlOc1_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[ir*(rdim+1)]*dev_opaddr2[im*(mdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_OlOc1(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc1_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Oc2
template <typename Tm>
__global__ void diagGPU_OlOc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[ir*(rdim+1)]*dev_opaddr2[iv*(vdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_OlOc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Ol*Or
template <typename Tm>
__global__ void diagGPU_OlOr_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[ir*(rdim+1)]*dev_opaddr2[ic*(cdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_OlOr(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_OlOr_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Oc2
template <typename Tm>
__global__ void diagGPU_Oc1Oc2_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[im*(mdim+1)]*dev_opaddr2[iv*(vdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_Oc1Oc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Oc2_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc1*Or
template <typename Tm>
__global__ void diagGPU_Oc1Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[im*(mdim+1)]*dev_opaddr2[ic*(cdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_Oc1Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   auto part = get_partition(nblk);
   dim3 dimBlock = part.first;
   dim3 dimGrid = part.second;
   diagGPU_Oc1Or_kernel<<<dimGrid,dimBlock>>>(nblk, dev_diag, dev_dims, 
         dev_opaddr1, dev_opaddr2, wt);
}

// Oc2*Or
template <typename Tm>
__global__ void diagGPU_Oc2Or_kernel(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if(i < nblk){
      size_t rdim = dev_dims[5*i];
      size_t cdim = dev_dims[5*i+1];
      size_t mdim = dev_dims[5*i+2];
      size_t vdim = dev_dims[5*i+3];
      size_t ircmv = dev_dims[5*i+4];
      for(int iv=0; iv<vdim; iv++){
         for(int im=0; im<mdim; im++){
            for(int ic=0; ic<cdim; ic++){
               for(int ir=0; ir<rdim; ir++){
                  dev_diag[ircmv] += wt*real_part(dev_opaddr1[iv*(vdim+1)]*dev_opaddr2[ic*(cdim+1)]);
                  ircmv++;
               } // ir
            } // ic
         } // im
      } // iv
   }
}
template <typename Tm>
void twodot_diagGPU_Oc2Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
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
   using Tm = std::complex<double>;
   Tm* dev_lopaddr = nullptr;
   Tm* dev_ropaddr = nullptr;
   Tm* dev_c1opaddr = nullptr;
   Tm* dev_c2opaddr = nullptr;
   ctns::twodot_diagGPU_local(nblk, dev_diag, dev_dims, dev_lopaddr, dev_ropaddr, dev_c1opaddr, dev_c2opaddr);
   ctns::twodot_diagGPU_OlOc1(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   ctns::twodot_diagGPU_OlOc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   ctns::twodot_diagGPU_OlOr(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   ctns::twodot_diagGPU_Oc1Oc2(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   ctns::twodot_diagGPU_Oc1Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   ctns::twodot_diagGPU_Oc2Or(nblk, dev_diag, dev_dims, dev_lopaddr, dev_c1opaddr);
   return 0;
}
*/
