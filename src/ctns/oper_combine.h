#ifndef OPER_COMBINE_H
#define OPER_COMBINE_H

#include <vector>
#include <tuple>

namespace ctns{

// combine support
inline std::vector<int> oper_combine_cindex(const std::vector<int>& cindex1,
		                            const std::vector<int>& cindex2){
   std::vector<int> cindex;
   cindex.insert(cindex.end(), cindex1.begin(), cindex1.end());
   cindex.insert(cindex.end(), cindex2.begin(), cindex2.end());
   return cindex;
}

inline void oper_combine_opC(const std::vector<int>& cindex1,
		      	     const std::vector<int>& cindex2,
		      	     std::vector<std::pair<int,int>>& info){
   int iop; 
   // 1. p1^+*I2
   iop = 1;
   for(int p1 : cindex1){
      info.push_back(std::make_pair(iop,p1));
   }
   // 2. I1*p2^+ 
   iop = 2;
   for(int p2 : cindex2){
      info.push_back(std::make_pair(iop,p2));
   }
}

// tricky part: determine the storage pattern for Apq
// nkr: only store Apq where p<q: total number (2K)*(2K-1)/2 ~ O(2K^2)
//  kr: time-reversal symmetry adapted basis is used, Apq blocks:
//      pA+qA+ and pA+qB+: K*(K-1)/2+K*(K+1)/2=K^2 (reduction by half)
inline void oper_combine_opA(const std::vector<int>& cindex1,
		             const std::vector<int>& cindex2,
		             const bool& ifkr,
		             std::vector<std::pair<int,int>>& info){
   int iop; 
   // 1. p1<q1: A[p1,q1] = p1^+q1^+ * I2
   iop = 1;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 < q1){
	    int index = oper_pack(p1,q1);
            info.push_back(std::make_pair(iop,index));
	    if(ifkr){
	       index = oper_pack(p1,q1+1);
               info.push_back(std::make_pair(iop,index));
	    }
	 }
      }
   }
   // 2. p2<q2: A[p2,q2] = I1 * p2^+q2^+ 
   iop = 2;
   for(int p2 : cindex2){
      for(int q2 : cindex2){
         if(p2 < q2){
	    int index = oper_pack(p2,q2);
            info.push_back(std::make_pair(iop,index));
	    if(ifkr){
	       index = oper_pack(p2,q2+1);
               info.push_back(std::make_pair(iop,index));
	    }
	 }
      }
   }
   // 3. p1<q2: A[p1,q2] = p1^+ * q2^+ 
   // 4. p1>q2: A[q2,p1] = q2^+ * p1^+ = -p1^+ * q2^+
   for(int p1 : cindex1){
      for(int q2 : cindex2){
	 iop = (p1<q2)? 3 : 4;
         int index = (p1<q2)? oper_pack(p1,q2) : oper_pack(q2,p1);
	 info.push_back(std::make_pair(iop,index));
         if(ifkr){ // Opposite-spin part:
	    index = (p1<q2)? oper_pack(p1,q2+1) : oper_pack(q2,p1+1);
	    info.push_back(std::make_pair(iop,index));
	 }
      }
   }
}

} // ctns

#endif
