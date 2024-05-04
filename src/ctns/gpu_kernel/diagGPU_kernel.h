#ifndef DIAGGPU_KERNEL_H
#define DIAGGPU_KERNEL_H

#include "cuComplex.h"
#define COMPLX cuDoubleComplex
#define COMPLX_MUL(a,b) cuCmul(a,b)

namespace ctns{

   const double thresh_diag_angular2 = 1.e-14;
   extern const double thresh_diag_angular2;

}

#endif
