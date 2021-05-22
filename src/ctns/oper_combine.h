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

inline std::vector<int> oper_gen_aindex(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> aindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 < q1){
	    int index = oper_pack(p1,q1);
            aindex.push_back(index);
	    if(ifkr){
	       index = oper_pack(p1,q1+1);
               aindex.push_back(index);
	    }
	 }
      }
   }
   int k = ifkr? cindex1.size() : cindex1.size()/2;
   int na = ifkr? k*k : k*(2*k-1);
   assert(aindex.size() == na);
   return aindex;
}

inline std::vector<int> oper_gen_bindex(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> bindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 <= q1){
	    int index = oper_pack(p1,q1);
            bindex.push_back(index);
	    if(ifkr){
	       index = oper_pack(p1,q1+1);
               bindex.push_back(index);
	    }
	 }
      }
   }
   int k = ifkr? cindex1.size() : cindex1.size()/2;
   int nb = ifkr? k*(k+1) : k*(2*k+1);
   assert(bindex.size() == nb);
   return bindex;
}

//	 
// Generate index for normal operators 
//	 

inline void oper_combine_opC(const std::vector<int>& cindex1,
		      	     const std::vector<int>& cindex2,
		      	     std::vector<std::pair<int,int>>& info){
   int iformula; 
   // 1. p1^+*I2
   iformula = 1;
   for(int p1 : cindex1){
      info.push_back(std::make_pair(iformula,p1));
   }
   // 2. I1*p2^+ 
   iformula = 2;
   for(int p2 : cindex2){
      info.push_back(std::make_pair(iformula,p2));
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
   int iformula; 
   // 1. p1<q1: A[p1,q1] = p1^+q1^+ * I2
   iformula = 1;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 < q1){
	    int index = oper_pack(p1,q1);
            info.push_back(std::make_pair(iformula,index));
	    if(ifkr){
	       index = oper_pack(p1,q1+1);
               info.push_back(std::make_pair(iformula,index));
	    }
	 }
      }
   }
   // 2. p2<q2: A[p2,q2] = I1 * p2^+q2^+ 
   iformula = 2;
   for(int p2 : cindex2){
      for(int q2 : cindex2){
         if(p2 < q2){
	    int index = oper_pack(p2,q2);
            info.push_back(std::make_pair(iformula,index));
	    if(ifkr){
	       index = oper_pack(p2,q2+1);
               info.push_back(std::make_pair(iformula,index));
	    }
	 }
      }
   }
   // 3. p1<q2: A[p1,q2] = p1^+ * q2^+ 
   // 4. p1>q2: A[q2,p1] = q2^+ * p1^+ = -p1^+ * q2^+
   for(int p1 : cindex1){
      for(int q2 : cindex2){
	 iformula = (p1<q2)? 3 : 4;
         int index = (p1<q2)? oper_pack(p1,q2) : oper_pack(q2,p1);
	 info.push_back(std::make_pair(iformula,index));
         if(ifkr){ // Opposite-spin part:
	    index = (p1<q2)? oper_pack(p1,q2+1) : oper_pack(q2,p1+1);
	    info.push_back(std::make_pair(iformula,index));
	 }
      }
   }
}

// tricky part: determine the storage pattern for Bps
// nkr: only store Bps (p<=s): (2K)*(2K+1)/2 ~ O(2K^2)
//  kr: If time-reversal symmetry adapted basis is used, Bps blocks:
//      pA+sA and pA+sB: K*(K+1)/2+K*(K+1)/2=K(K+1) (reduction by half)
inline void oper_combine_opB(const std::vector<int>& cindex1,
		             const std::vector<int>& cindex2,
		             const bool& ifkr,
		             std::vector<std::pair<int,int>>& info){
   int iformula; 
   // 1. p1<q1: B[p1,q1] = p1^+q1 * I2
   iformula = 1;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 <= q1){
	    int index = oper_pack(p1,q1);
            info.push_back(std::make_pair(iformula,index));
	    if(ifkr){
	       index = oper_pack(p1,q1+1);
               info.push_back(std::make_pair(iformula,index));
	    }
	 }
      }
   }
   // 2. p2<=q2: B[p2,q2] = I1 * p2^+q2
   iformula = 2;
   for(int p2 : cindex2){
      for(int q2 : cindex2){
         if(p2 <= q2){
	    int index = oper_pack(p2,q2);
            info.push_back(std::make_pair(iformula,index));
	    if(ifkr){
	       index = oper_pack(p2,q2+1);
               info.push_back(std::make_pair(iformula,index));
	    }
	 }
      }
   }
   // 3. p1<q2: B[p1,q2] = p1^+ * q2
   // 4. p1>q2: B[q2,p1] = q2 * p1^+ = -p1^+ * q2
   for(int p1 : cindex1){
      for(int q2 : cindex2){
	 iformula = (p1<q2)? 3 : 4;
         int index = (p1<q2)? oper_pack(p1,q2) : oper_pack(q2,p1);
	 info.push_back(std::make_pair(iformula,index));
         if(ifkr){ // Opposite-spin part:
	    index = (p1<q2)? oper_pack(p1,q2+1) : oper_pack(q2,p1+1);
	    info.push_back(std::make_pair(iformula,index));
	 }
      }
   }
}

//
// Generate index for complementary operators 
//	 

// tricky part: determine the storage pattern for Ppq for p,q in krest
inline void oper_combine_opP(const std::vector<int>& krest,
		             const bool& ifkr,
		             std::vector<int>& info){
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int kq : krest){
	 int qa = 2*kq, qb = qa+1;
	 if(kp < kq){
            info.push_back(oper_pack(pa,qa)); // Paa 
	    info.push_back(oper_pack(pa,qb)); // Pab
	    if(not ifkr){
	       // since if kp<kq, pb<qa and pb<qb hold
	       info.push_back(oper_pack(pb,qa));
	       info.push_back(oper_pack(pb,qb));
	    }
	 }else if(kp == kq){
            info.push_back(oper_pack(pa,pb)); // Pab 
	 }
      } // kq
   } // kp
}

inline void oper_combine_opQ(const std::vector<int>& krest,
		             const bool& ifkr,
		             std::vector<int>& info){
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int ks : krest){
	 int sa = 2*ks, sb = sa+1;
	 if(kp <= ks){ 
	    info.push_back(oper_pack(pa,sa));
	    info.push_back(oper_pack(pa,sb));
	    if(not ifkr){
	       // if kp=ks, QpApB is stored while QpBpA is redundant,
	       // because it can be related with QpApB using Hermiticity if bra=ket.
	       if(kp != ks) info.push_back(oper_pack(pb,sa));
	       info.push_back(oper_pack(pb,sb));
	    }
	 }
      }
   }
}

inline void oper_combine_opS(const std::vector<int>& krest,
		             const bool& ifkr,
		             std::vector<int>& info){
   for(int kp: krest){
      int pa = 2*kp, pb = pa+1;
      info.push_back(pa);
      if(not ifkr) info.push_back(pb);
   }
}

} // ctns

#endif
