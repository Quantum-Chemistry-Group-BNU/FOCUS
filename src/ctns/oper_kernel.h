#ifndef OPER_KERNEL_H
#define OPER_KERNEL_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include "qtensor/qtensor.h"
#include "oper_timer.h"

namespace ctns{

template <typename Tm>
using cqt3qt2 = std::function<stensor3<Tm>(const stensor3<Tm>&, const stensor2<Tm>&)>;

// O1*I2|psi> 
template <typename Tm> 
stensor3<Tm> oper_kernel_OIwf(const std::string& superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o1,
			      const bool ifdagger=false){
   std::map<char,cqt3qt2<Tm>> fdict = {{'l',&contract_qt3_qt2_l<Tm>}, 
	    	   	 	       {'c',&contract_qt3_qt2_c<Tm>},
	     	       		       {'r',&contract_qt3_qt2_r<Tm>}};
   stensor3<Tm> qt3;
   if(ifdagger){
      qt3 = fdict[superblock[0]](ksite, o1.H());
   }else{
      qt3 = fdict[superblock[0]](ksite, o1);
   }
   return qt3;
}

// I1*O2|psi> 
template <typename Tm> 
stensor3<Tm> oper_kernel_IOwf(const std::string& superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o2,
			      const bool po2, // parity of O2
			      const bool ifdagger=false){
   std::map<char,cqt3qt2<Tm>> fdict = {{'l',&contract_qt3_qt2_l<Tm>}, 
	    	   	 	       {'c',&contract_qt3_qt2_c<Tm>},
	     	       		       {'r',&contract_qt3_qt2_r<Tm>}};
   stensor3<Tm> qt3;
   if(ifdagger){
      qt3 = fdict[superblock[1]](ksite, o2.H());
   }else{
      qt3 = fdict[superblock[1]](ksite, o2);
   }
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
stensor3<Tm> oper_kernel_OOwf(const std::string& superblock,
			      const stensor3<Tm>& ksite,
 			      const stensor2<Tm>& o1,
 			      const stensor2<Tm>& o2,
			      const bool po2,
			      const bool ifdagger=false){
   auto qt3 = oper_kernel_IOwf(superblock, ksite, o2, po2, ifdagger);
   return oper_kernel_OIwf(superblock, qt3, o1, ifdagger);
}

// <bra|[O|ket>]
template <typename Tm> 
stensor2<Tm> oper_kernel_renorm(const std::string& superblock,
			        const stensor3<Tm>& bsite,
				const stensor3<Tm>& ksite){
   auto t0 = tools::get_time();

   stensor2<Tm> qt2;
   if(superblock == "lc"){
      qt2 = contract_qt3_qt3_lc(bsite,ksite);
   }else if(superblock == "lr"){
      qt2 = contract_qt3_qt3_lr(bsite,ksite);
   }else if(superblock == "cr"){
      qt2 = contract_qt3_qt3_cr(bsite,ksite);
   }else{
      std::string msg = "error: no such case in oper_kernel_renorm!";
      tools::exit(msg+" superblock="+superblock);
   }

   auto t1 = tools::get_time();
#ifdef _OPENMP
   #pragma omp critical
#endif
   {
      oper_timer.nR += 1;
      oper_timer.tR += tools::get_duration(t1-t0);
   }
   return qt2;
}

} // ctns

#endif
