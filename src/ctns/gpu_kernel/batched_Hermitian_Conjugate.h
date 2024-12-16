#ifndef BATCHED_HERMITIAN_CONJUGATE_H
#define BATCHED_HERMITIAN_CONJUGATE_H

#include "ctnsGPU.h"

namespace ctns{

   // H[loc] 
   template <typename Tm>
      void batched_Hermitian_Conjugate(const size_t nblks,
            const size_t* dev_offs,
            const int* dev_dims,
            const Tm* dev_facs,
            const Tm* dev_qops1,
            Tm* dev_qops2){
         exit(1);
      }
   template <>
      void batched_Hermitian_Conjugate(const size_t nblks,
            const size_t* dev_offs,
            const int* dev_dims,
            const double* dev_facs,
            const double* dev_qops1,
            double* dev_qops2);
   template <>
      void batched_Hermitian_Conjugate(const size_t nblks,
            const size_t* dev_offs,
            const int* dev_dims,
            const COMPLX* dev_facs,
            const COMPLX* dev_qops1,
            COMPLX* dev_qops2);

}

#endif
