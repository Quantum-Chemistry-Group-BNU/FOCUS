#ifndef TWODOT_DIAGGPU_KERNEL_H
#define TWODOT_DIAGGPU_KERNEL_H

namespace ctns{

// H[loc] 
template <typename Tm>
void twodot_diagGPU_local(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_lopaddr,
      const Tm* dev_ropaddr,
      const Tm* dev_c1opaddr,
      const Tm* dev_c2opaddr);

// OlOc1
template <typename Tm>
void twodot_diagGPU_OlOc1(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

// OlOc2
template <typename Tm>
void twodot_diagGPU_OlOc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

// OlOr
template <typename Tm>
void twodot_diagGPU_OlOr(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

// Oc1Oc2
template <typename Tm>
void twodot_diagGPU_Oc1Oc2(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

// Oc1Or
template <typename Tm>
void twodot_diagGPU_Oc1Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

// Oc2Or
template <typename Tm>
void twodot_diagGPU_Oc2Or(const size_t nblk,
      double* dev_diag,
      const size_t* dev_dims,
      const Tm* dev_opaddr1,
      const Tm* dev_opaddr2,
      const double wt);

}

#endif
