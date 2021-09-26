#ifndef OPER_KERNEL_H
#define OPER_KERNEL_H

#include "qtensor/qtensor.h"

namespace ctns{

// O1*I2|psi> 
template <typename Tm> 
stensor3<Tm> oper_kernel_OIwf(const std::string superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o1,
			      const bool ifdagger=false){
   return contract_qt3_qt2(superblock[0], ksite, o1, ifdagger);
}

// I1*O2|psi> 
template <typename Tm> 
stensor3<Tm> oper_kernel_IOwf(const std::string superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o2,
			      const bool po2, // parity of O2
			      const bool ifdagger=false){
   auto qt3 = contract_qt3_qt2(superblock[1], ksite, o2, ifdagger);
   // Il*Oc|psi> => (-1)^{p(l)}Oc[c',c]psi[l,c,r]
   if(po2 && superblock == "lc") qt3.row_signed();
   if(po2 && superblock == "lr") qt3.row_signed();
   // Ic*Or|psi> => (-1)^{p(c)}Or[r',r']psi[l,c,r]
   if(po2 && superblock == "cr") qt3.mid_signed();
   return qt3;
}

// O1^d*O2^d|psi>: Note that it differs from (O1*O2)^d. 
// The possible sign change needs to be taken into account outside this function.
template <typename Tm> 
stensor3<Tm> oper_kernel_OOwf(const std::string superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o1,
 			      const stensor2<Tm>& o2,
			      const bool po2,
			      const bool ifdagger=false){
   auto qt3 = oper_kernel_IOwf(superblock, ksite, o2, po2, ifdagger);
   return oper_kernel_OIwf(superblock, qt3, o1, ifdagger);
}

} // ctns

#endif
