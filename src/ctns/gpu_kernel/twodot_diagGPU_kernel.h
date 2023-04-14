#ifndef TWODOT_DIAGGPU_KERNEL_H
#define TWODOT_DIAGGPU_KERNEL_H

#include "cuComplex.h"
#define COMPLX cuDoubleComplex
#define COMPLX_MUL(a,b) cuCmul(a,b)

namespace ctns{

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
   twodot_diagGPU_local(nblk, ndim, dev_diag, dev_dims, 
         (COMPLX*)dev_lopaddr,
         (COMPLX*)dev_ropaddr,
         (COMPLX*)dev_c1opaddr,
         (COMPLX*)dev_c2opaddr);
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

// OlOc1
template <typename Tm>
void twodot_diagGPU_OlOc1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_OlOc1(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1, 
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_OlOc1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_OlOc1(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

// OlOc2
template <typename Tm>
void twodot_diagGPU_OlOc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_OlOc2(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1,
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_OlOc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_OlOc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

// OlOr
template <typename Tm>
void twodot_diagGPU_OlOr(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_OlOr(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1,
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_OlOr(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_OlOr(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

// Oc1Oc2
template <typename Tm>
void twodot_diagGPU_Oc1Oc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_Oc1Oc2(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1,
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_Oc1Oc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_Oc1Oc2(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

// Oc1Or
template <typename Tm>
void twodot_diagGPU_Oc1Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_Oc1Or(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1,
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_Oc1Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_Oc1Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

// Oc2Or
template <typename Tm>
void twodot_diagGPU_Oc2Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt){
   twodot_diagGPU_Oc2Or(nblk, ndim, dev_diag, dev_dims,
         (COMPLX*)dev_opaddr1,
         (COMPLX*)dev_opaddr2,
         wt);
}
template <>
void twodot_diagGPU_Oc2Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const double* dev_opaddr1,
      const double* dev_opaddr2,
      const double wt);
template <>
void twodot_diagGPU_Oc2Or(const size_t nblk,
      const size_t ndim,
      double* dev_diag,
      const size_t* dev_dims,
      const COMPLX* dev_opaddr1,
      const COMPLX* dev_opaddr2,
      const double wt);

}

#endif
