#ifndef OPER_KERNEL_H
#define OPER_KERNEL_H

#include "qtensor.h"

namespace ctns{

// O1*I2|psi> 
template <typename Tm> 
qtensor3<Tm> oper_kernel_OIwf(const std::string& superblock,
			      const qtensor3<Tm>& ksite,
 			      const qtensor2<Tm>& o1,
			      const bool ifdagger=false){
   qtensor3<Tm> qt3;
   if(superblock == "lc" || superblock == "lr"){
      if(ifdagger){
         qt3 = contract_qt3_qt2_l(ksite,o1.H());
      }else{
         qt3 = contract_qt3_qt2_l(ksite,o1);
      }
   }else if(superblock == "cr"){
      if(ifdagger){
         qt3 = contract_qt3_qt2_c(ksite,o1.H());
      }else{
         qt3 = contract_qt3_qt2_c(ksite,o1);
      }
   }else{
      std::cout << "error: no such case in oper_kernel_OIwf!"
	        << " superblock=" << superblock << std::endl;
      exit(1);
   }
   return qt3;
}

// I1*O2|psi> 
template <typename Tm> 
qtensor3<Tm> oper_kernel_IOwf(const std::string& superblock,
			      const qtensor3<Tm>& ksite,
 			      const qtensor2<Tm>& o2,
			      const bool po2, // parity of O2
			      const bool ifdagger=false){
   qtensor3<Tm> qt3;
   if(superblock == "lc"){
      if(ifdagger){
	 qt3 = contract_qt3_qt2_c(ksite,o2.H());
      }else{
	 qt3 = contract_qt3_qt2_c(ksite,o2);
      }
      // Il*Oc|psi> => (-1)^{p(l)}Oc[c',c]psi[l,c,r]
      if(po2) qt3 = qt3.row_signed();
   }else if(superblock == "lr"){
      if(ifdagger){
         qt3 = contract_qt3_qt2_r(ksite,o2.H());
      }else{
         qt3 = contract_qt3_qt2_r(ksite,o2);
      }
      if(po2) qt3 = qt3.row_signed();
   }else if(superblock == "cr"){
      if(ifdagger){
	 qt3 = contract_qt3_qt2_r(ksite,o2.H());
      }else{
	 qt3 = contract_qt3_qt2_r(ksite,o2);
      }
      // Ic*Or|psi> => (-1)^{p(c)}Or[r',r']psi[l,c,r]
      if(po2) qt3 = qt3.mid_signed();
   }else{
      std::cout << "error: no such case in oper_kernel_IOwf!"
	        << " superblock=" << superblock << std::endl;
      exit(1);
   }
   return qt3;
}

// O1*O2|psi>
template <typename Tm> 
qtensor3<Tm> oper_kernel_OOwf(const std::string& superblock,
			      const qtensor3<Tm>& ksite,
 			      const qtensor2<Tm>& o1,
 			      const qtensor2<Tm>& o2,
			      const bool po2,
			      const bool ifdagger=false){
   auto qt3 = oper_kernel_IOwf(superblock, ksite, o2, po2, ifdagger);
   return oper_kernel_OIwf(superblock, qt3, o1, ifdagger);
}

// <bra|[O|ket>]
template <typename Tm> 
qtensor2<Tm> oper_kernel_renorm(const std::string& superblock,
			        const qtensor3<Tm>& bsite,
				const qtensor3<Tm>& ksite){
   qtensor2<Tm> qt2;
   if(superblock == "lc"){
      qt2 = contract_qt3_qt3_lc(bsite,ksite);
   }else if(superblock == "lr"){
      qt2 = contract_qt3_qt3_lr(bsite,ksite);
   }else if(superblock == "cr"){
      qt2 = contract_qt3_qt3_cr(bsite,ksite);
   }else{
      std::cout << "error: no such case in oper_kernel_renorm!"
	        << " superblock=" << superblock << std::endl;
      exit(1);
   }
   return qt2;
}

} // ctns

#endif
