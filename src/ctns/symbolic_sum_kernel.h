#ifndef SYMBOLIC_SUM_KERNEL_H
#define SYMBOLIC_SUM_KERNEL_H

#include "oper_dict.h"
#include "symbolic_oper.h"

namespace ctns{

template <typename Tm>
stensor2<Tm> symbolic_sum_oper(const oper_dict<Tm>& qops,
			       const symbolic_sum<Tm>& sop,
	          	       const char& label,
			       const bool& dagger,
			       Tm* workspace){
   int len = sop.size();
   // we assume the rest of terms have the same label/dagger
   auto wt0 = sop.sums[0].first;
   const auto& sop0 = sop.sums[0].second;
   int index0 = sop0.index;
   int nbar0  = sop0.nbar;
   // form opsum = wt0*op0 + wt1*op1 + ...
   const auto& op0 = qops(label).at(index0);
   if(dagger) wt0 = tools::conjugate(wt0);
   stensor2<Tm> optmp;
   optmp.init(op0.info,false);
   optmp.setup_data(workspace);
   optmp.clear();
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
   if(dagger) linalg::xconj(optmp.size(), optmp.data());
   return optmp;
}

} // ctns	

#endif
