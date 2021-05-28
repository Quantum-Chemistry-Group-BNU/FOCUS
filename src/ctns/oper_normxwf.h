#ifndef OPER_NORMXWF_H
#define OPER_NORMXWF_H

#include "oper_kernel.h"

namespace ctns{

// kernel for computing Cp|ket> 
template <typename Tm>
qtensor3<Tm> oper_normxwf_opC(const std::string& superblock,
		              const qtensor3<Tm>& site,
		              oper_dict<Tm>& qops1,
		              oper_dict<Tm>& qops2,
			      const int iformula,
		              const int index,
			      const bool ifdagger=false){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('C')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1,ifdagger);
   }else if(iformula == 2){
      const auto& op2 = qops2('C')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,1,ifdagger);
   } // iformula
   return opwf; 
}

// kernel for computing Apq|ket> 
template <typename Tm>
qtensor3<Tm> oper_normxwf_opA(const std::string& superblock,
		              const qtensor3<Tm>& site,
		              oper_dict<Tm>& qops1,
		              oper_dict<Tm>& qops2,
			      const bool& ifkr,
			      const int iformula,
		              const int index,
			      const bool ifdagger=false){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('A')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1,ifdagger);
   }else if(iformula == 2){
      const auto& op2 = qops2('A')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,0,ifdagger);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // kr opposite spin case: <a1A^+a2B^+> = [a1A^+]*[a2B^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = qops1('C')[p];
      const auto& op2 = ifnot_kros? qops2('C')[q] : qops2('C')[q-1].K(1);
      opwf = oper_kernel_OOwf(superblock,site,op1,op2,1,ifdagger);
      if(ifdagger) opwf = -opwf; // (c1*c2)^d = c2d*c1d = -c1d*c2d
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // kr opposite spin case: <a2A^+a1B^+> = -[a1B^+]*[a2A^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = ifnot_kros? -qops1('C')[p] : -qops1('C')[p-1].K(1);
      const auto& op2 = qops2('C')[q];
      opwf = oper_kernel_OOwf(superblock,site,op1,op2,1,ifdagger);
      if(ifdagger) opwf = -opwf;
   } // iformula
   return opwf;
}

// kernel for computing Bps|ket> 
template <typename Tm>
qtensor3<Tm> oper_normxwf_opB(const std::string& superblock,
		              const qtensor3<Tm>& site,
		              oper_dict<Tm>& qops1,
		              oper_dict<Tm>& qops2,
			      const bool& ifkr,
			      const int iformula,
		              const int index,
			      const bool ifdagger=false){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('B')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1,ifdagger);
   }else if(iformula == 2){
      const auto& op2 = qops2('B')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,0,ifdagger);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // kr opposite spin case: <a1A^+a2B> = [a1A^+]*[a2B]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = qops1('C')[p];
      const auto& op2 = ifnot_kros? qops2('C')[q].H() : qops2('C')[q-1].H().K(1);
      opwf = oper_kernel_OOwf(superblock,site,op1,op2,1,ifdagger);
      if(ifdagger) opwf = -opwf;
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // kr opposite spin case: <a2A^+a1B> = -[a1B]*[a2A^+]
      const bool ifnot_kros = !(ifkr && sp != sq);
      const auto& op1 = ifnot_kros? -qops1('C')[p].H() : -qops1('C')[p-1].H().K(1);
      const auto& op2 = qops2('C')[q];
      opwf = oper_kernel_OOwf(superblock,site,op1,op2,1,ifdagger);
      if(ifdagger) opwf = -opwf;
   } // iformula
   return opwf;
}

} // ctns

#endif
