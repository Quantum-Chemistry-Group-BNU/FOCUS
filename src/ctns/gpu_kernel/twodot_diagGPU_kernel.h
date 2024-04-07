#ifndef TWODOT_DIAGGPU_KERNEL_H
#define TWODOT_DIAGGPU_KERNEL_H

#include "cuComplex.h"
#define COMPLX cuDoubleComplex
#define COMPLX_MUL(a,b) cuCmul(a,b)

namespace ctns{

   const double thresh_diag_angular2 = 1.e-14;
   extern const double thresh_diag_angular2;

   // H[loc] 
   template <typename Tm>
      void twodot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_lopaddr,
            const Tm* dev_ropaddr,
            const Tm* dev_c1opaddr,
            const Tm* dev_c2opaddr){
         exit(1);
         //twodot_diagGPU_local(nblk, ndim, dev_diag, dev_dims, 
         //      (COMPLX*)dev_lopaddr,
         //      (COMPLX*)dev_ropaddr,
         //      (COMPLX*)dev_c1opaddr,
         //      (COMPLX*)dev_c2opaddr);
      }
   template <>
      void twodot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_lopaddr,
            const double* dev_ropaddr,
            const double* dev_c1opaddr,
            const double* dev_c2opaddr);
   template <>
      void twodot_diagGPU_local(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const COMPLX* dev_lopaddr,
            const COMPLX* dev_ropaddr,
            const COMPLX* dev_c1opaddr,
            const COMPLX* dev_c2opaddr);

   // O1O2
   template <typename Tm>
      void twodot_diagGPU_O1O2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_opaddr1,
            const Tm* dev_opaddr2,
            const double wt,
            const int i1,
            const int i2){
         exit(1);
         //twodot_diagGPU_O1O2(nblk, ndim, dev_diag, dev_dims,
         //      (COMPLX*)dev_opaddr1, 
         //      (COMPLX*)dev_opaddr2,
         //      wt, i1, i2);
      }
   template <>
      void twodot_diagGPU_O1O2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_opaddr1,
            const double* dev_opaddr2,
            const double wt,
            const int i1,
            const int i2);
   template <>
      void twodot_diagGPU_O1O2(const size_t nblk,
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
      void twodot_diagGPU_O1O2_su2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const Tm* dev_opaddr1,
            const Tm* dev_opaddr2,
            const double* dev_fac,
            const int i1,
            const int i2){
         exit(1);
         //twodot_diagGPU_O1O2_su2(nblk, ndim, dev_diag, dev_dims,
         //      (COMPLX*)dev_opaddr1, 
         //      (COMPLX*)dev_opaddr2,
         //      dev_fac, i1, i2);
      }
   template <>
      void twodot_diagGPU_O1O2_su2(const size_t nblk,
            const size_t ndim,
            double* dev_diag,
            const size_t* dev_dims,
            const double* dev_opaddr1,
            const double* dev_opaddr2,
            const double* dev_fac,
            const int i1,
            const int i2);
   template <>
      void twodot_diagGPU_O1O2_su2(const size_t nblk,
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
