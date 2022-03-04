#ifndef SYMBOLIC_NORMXWF_H
#define SYMBOLIC_NORMXWF_H

#include "symbolic_oper.h"

namespace ctns{

// kernel for computing Cp|ket>
template <typename Tm>
symbolic_task<Tm> symbolic_normxwf_opC(const std::string block1,
				       const std::string block2,
			               const int index,
			               const int iformula,
		                       const bool ifdagger=false){
   symbolic_task<Tm> formulae;
   if(iformula == 1){
      auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'C',index,ifdagger));
      formulae.append(op1);
   }else if(iformula == 2){
      auto op2 = symbolic_prod<Tm>(symbolic_oper(block2,'C',index,ifdagger));
      formulae.append(op2);
   } // iformula
   return formulae;
}

// kernel for computing Apq|ket> 
template <typename Tm>
symbolic_task<Tm> symbolic_normxwf_opA(const std::string block1,
				       const std::string block2,
		                       const int index,
			               const int iformula,
				       const bool ifkr,
			               const bool ifdagger=false){
   symbolic_task<Tm> formulae;
   if(iformula == 1){
      // A[p1q1]
      auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'A',index,ifdagger));
      formulae.append(op1);
   }else if(iformula == 2){
      // A[p2q2]
      auto op2 = symbolic_prod<Tm>(symbolic_oper(block2,'A',index,ifdagger));
      formulae.append(op2);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // A[p1<q2] = p1+q2+
      auto op1 = symbolic_oper(block1,'C',p,ifdagger);
      const bool ifnot_kros = !(ifkr && sp != sq);
      auto op2 = ifnot_kros? symbolic_oper(block2,'C',q,ifdagger) : 
	      		     symbolic_oper(block2,'C',q-1,ifdagger).K(1);
      auto op12 = symbolic_prod<Tm>(op1,op2);
      // (c1*c2)^d = c2d*c1d = -c1d*c2d
      if(ifdagger) op12.scale(-1.0);
      formulae.append(op12);
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // A[q2<p1] = q2+p1+ = -p1+q2+
      const bool ifnot_kros = !(ifkr && sp != sq);
      auto op1 = ifnot_kros? symbolic_oper(block1,'C',p,ifdagger) :
	      	             symbolic_oper(block1,'C',p-1,ifdagger).K(1);
      auto op2 = symbolic_oper(block2,'C',q,ifdagger);
      auto op12 = symbolic_prod<Tm>(op1,op2,-1.0);
      if(ifdagger) op12.scale(-1.0);
      formulae.append(op12);
   } // iformula
   return formulae;
}

// kernel for computing Bps|ket>
template <typename Tm>
symbolic_task<Tm> symbolic_normxwf_opB(const std::string block1,
				       const std::string block2,
		              	       const int index,
			      	       const int iformula,
				       const bool ifkr,
			      	       const bool ifdagger=false){
   symbolic_task<Tm> formulae;
   if(iformula == 1){
      // B[p1q1]
      auto op1 = symbolic_prod<Tm>(symbolic_oper(block1,'B',index,ifdagger));
      formulae.append(op1);
   }else if(iformula == 2){
      // B[p2q2]
      auto op1 = symbolic_prod<Tm>(symbolic_oper(block2,'B',index,ifdagger));
      formulae.append(op1);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      // B[p1q2] = p1+q2
      auto op1 = symbolic_oper(block1,'C',p,ifdagger);
      const bool ifnot_kros = !(ifkr && sp != sq);
      auto op2 = ifnot_kros? symbolic_oper(block2,'C',q,!ifdagger) :
	      		     symbolic_oper(block2,'C',q-1,!ifdagger).K(1);
      auto op12 = symbolic_prod<Tm>(op1,op2);
      if(ifdagger) op12.scale(-1.0);
      formulae.append(op12);
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      // B[q2p1] = q2+p1 = -p1q2+
      const bool ifnot_kros = !(ifkr && sp != sq);
      auto op1 = ifnot_kros? symbolic_oper(block1,'C',p,!ifdagger) : 
	      	 	     symbolic_oper(block1,'C',p-1,!ifdagger).K(1);
      auto op2 = symbolic_oper(block2,'C',q,ifdagger);
      auto op12 = symbolic_prod<Tm>(op1,op2,-1.0);
      if(ifdagger) op12.scale(-1.0);
      formulae.append(op12);
   } // iformula
   return formulae;
}

} // ctns

#endif
