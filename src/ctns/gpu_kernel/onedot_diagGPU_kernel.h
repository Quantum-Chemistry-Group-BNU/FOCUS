#ifndef ONEDOT_DIAGGPU_KERNEL_H
#define ONEDOT_DIAGGPU_KERNEL_H

#include "ctnsGPU.h"

namespace ctns{

   // H[loc] 
   template <typename Tm>
      void onedot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_lopaddr,
            const Tm* dev_ropaddr,
            const Tm* dev_copaddr){
         exit(1);
      }
   template <>
      void onedot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_lopaddr,
            const double* dev_ropaddr,
            const double* dev_copaddr);
   template <>
      void onedot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const COMPLX* dev_lopaddr,
            const COMPLX* dev_ropaddr,
            const COMPLX* dev_copaddr);

   // O1O2
   template <typename Tm>
      void onedot_diagGPU_O1O2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_opaddr1,
            const Tm* dev_opaddr2,
            const double wt,
            const int i1,
            const int i2){
         exit(1);
      }
   template <>
      void onedot_diagGPU_O1O2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_opaddr1,
            const double* dev_opaddr2,
            const double wt,
            const int i1,
            const int i2);
   template <>
      void onedot_diagGPU_O1O2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const COMPLX* dev_opaddr1,
            const COMPLX* dev_opaddr2,
            const double wt,
            const int i1,
            const int i2);

   // O1O2_su2
   template <typename Tm>
      void onedot_diagGPU_O1O2_su2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_opaddr1,
            const Tm* dev_opaddr2,
            const double* dev_fac,
            const int i1,
            const int i2){
         exit(1);
      }
   template <>
      void onedot_diagGPU_O1O2_su2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_opaddr1,
            const double* dev_opaddr2,
            const double* dev_fac,
            const int i1,
            const int i2);
   template <>
      void onedot_diagGPU_O1O2_su2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const COMPLX* dev_opaddr1,
            const COMPLX* dev_opaddr2,
            const double* dev_fac,
            const int i1,
            const int i2);

}

#endif
