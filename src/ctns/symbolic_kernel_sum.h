#ifndef SYMBOLIC_KERNEL_SUM_H
#define SYMBOLIC_KERNEL_SUM_H

#include "oper_dict.h"
#include "symbolic_oper.h"

namespace ctns{

// formulae = (w a^d) => (w^d a)^d, return w^d a 
template <typename Tm>
stensor2<Tm> symbolic_sum_oper(const oper_dictmap<Tm>& qops_dict,
			       const symbolic_sum<Tm>& sop,
			       Tm* workspace){
   int len = sop.size();
   // we assume the rest of terms have the same label/dagger
   auto wt0 = sop.sums[0].first;
   const auto& sop0 = sop.sums[0].second;
   const auto& block = sop0.block;
   const auto& label = sop0.label;
   const auto& dagger= sop0.dagger;
   int index0 = sop0.index;
   int nbar0  = sop0.nbar;
   // form opsum = wt0*op0 + wt1*op1 + ...
   const auto& qops = qops_dict.at(block);
   const auto& op0 = qops(label).at(index0);
   if(dagger) wt0 = tools::conjugate(wt0);
   stensor2<Tm> optmp;
   optmp.init(op0.info,false);
   optmp.setup_data(workspace);
   optmp.set_zero();
   if(nbar0 == 0){
      linalg::xaxpy(op0.size(), wt0, op0.data(), optmp.data());
   }else{
      auto op0k = op0.K(nbar0);
      linalg::xaxpy(op0.size(), wt0, op0k.data(), optmp.data());
   }
   for(int k=1; k<len; k++){
      auto wtk = sop.sums[k].first;
      const auto& sopk = sop.sums[k].second;
      int indexk = sopk.index;
      int nbark  = sopk.nbar;
      const auto& opk = qops(label).at(indexk);
      if(dagger) wtk = tools::conjugate(wtk);
      if(nbark == 0){
         linalg::xaxpy(opk.size(), wtk, opk.data(), optmp.data());
      }else{
         auto opkk = opk.K(nbark);
         linalg::xaxpy(opk.size(), wtk, opkk.data(), optmp.data());
      }
   } // k
   /*
   std::cout << "sop=" << sop << std::endl; 
   optmp.print("optmp",2);
   */
   return optmp;
}

} // ctns	

#endif
