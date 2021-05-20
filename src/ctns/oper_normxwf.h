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
		              const int index){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('C')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1);
   }else if(iformula == 2){
      const auto& op2 = qops2('C')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,1);
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
		              const int index){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('A')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1);
   }else if(iformula == 2){
      const auto& op2 = qops2('A')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,0);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      const auto& op1 = qops1('C')[p];
      const auto& op2 = qops2('C')[q];
      if(not ifkr && (ifkr && sp == sq)){
         opwf = oper_kernel_OOwf(superblock,site,op1,op2,1);
      }else{
         // kr opposite spin case: <a1A^+a2B^+> = [a1^+]*[a2^+]
         opwf = oper_kernel_OOwf(superblock,site,op1,op2.K(1),1);
      }
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      const auto& op1 = -qops1('C')[p];
      const auto& op2 = qops2('C')[q];
      if(not ifkr && (ifkr && sp == sq)){
         opwf = oper_kernel_OOwf(superblock,site,op1,op2,1);
      }else{
         // kr opposite spin case: <a2A^+a1B^+> = -[a1^+]*[a2^+]
         opwf = oper_kernel_OOwf(superblock,site,op1.K(1),op2,1);
      }
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
		              const int index){
   qtensor3<Tm> opwf;
   if(iformula == 1){
      const auto& op1 = qops1('B')[index];
      opwf = oper_kernel_OIwf(superblock,site,op1);
   }else if(iformula == 2){
      const auto& op2 = qops2('B')[index];
      opwf = oper_kernel_IOwf(superblock,site,op2,0);
   }else if(iformula == 3){
      auto pq = oper_unpack(index);	
      int p = pq.first, sp = p%2;
      int q = pq.second, sq = q%2;
      const auto& op1 = qops1('C')[p];
      const auto& op2 = qops2('C')[q].H();
      if(not ifkr && (ifkr && sp == sq)){
         opwf = oper_kernel_OOwf(superblock,site,op1,op2,1);
      }else{
         // kr opposite spin case: <a1A^+a2B> = [a1^+]*[a2^+]
         opwf = oper_kernel_OOwf(superblock,site,op1,op2.K(1),1);
      }
   }else if(iformula == 4){
      auto qp = oper_unpack(index);	
      int p = qp.second, sp = p%2;
      int q = qp.first, sq = q%2;
      const auto& op1 = -qops1('C')[p].H();
      const auto& op2 = qops2('C')[q];
      if(not ifkr && (ifkr && sp == sq)){
         opwf = oper_kernel_OOwf(superblock,site,op1,op2,1);
      }else{
         // kr opposite spin case: <a2A^+a1B> = -[a1^+]*[a2^+]
         opwf = oper_kernel_OOwf(superblock,site,op1.K(1),op2,1);
      }
   } // iformula
   return opwf;
}

} // ctns

#endif
