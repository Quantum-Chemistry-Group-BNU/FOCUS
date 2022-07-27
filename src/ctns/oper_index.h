#ifndef OPER_INDEX_H
#define OPER_INDEX_H

namespace ctns{

// --- packing (i,j) into ij ---
const int kpack = 1000;
extern const int kpack;
// pack & unpack
inline int oper_pack(const int i, const int j){ 
   return i+j*kpack;
}
inline std::pair<int,int> oper_unpack(const int ij){
   return std::make_pair(ij%kpack,ij/kpack);
}

// --- weight factor in H ---
// ifkr = false
inline double wfac(const int ij){
   int i = ij%kpack;
   int j = ij/kpack;
   return (i==j)? 0.5 : 1.0; 
}
// ifkr = true
inline double wfacAP(const int ij){
   int i = ij%kpack, ki = i/2, spin_i = i%2;
   int j = ij/kpack, kj = j/2, spin_j = j%2;
   if(spin_i == spin_j){
      return 1.0;
   }else{
      return (ki==kj)? 0.5 : 1.0; // avoid duplication for A[p\bar{p}]
   }
}
inline double wfacBQ(const int ij){
   int i = ij%kpack, ki = i/2, spin_i = i%2;
   int j = ij/kpack, kj = j/2, spin_j = j%2;
   return (ki==kj)? 0.5 : 1.0;
}

// --- distribution of p or (p,q) for MPI ---
inline int distribute1(const bool ifkr, const int size, const int index){
   // only SpA is kept for ifkr=true
   return ifkr? (index/2)%size : index%size; 
}

inline int distribute2(const bool ifkr, const int size, const int index){
   auto pq = oper_unpack(index);
   int p = pq.first;
   int q = pq.second;
   assert(p <= q);
   return (q*(q+1)/2+p)%size;
}

inline std::vector<int> distribute2(const bool ifkr, const int size, 
			            const std::vector<int>& index_full,
				    const int rank){
   std::vector<int> index_dist;
   for(int idx : index_full){
      int iproc = distribute2(ifkr, size, idx);
      if(iproc == rank) index_dist.push_back(idx);
   }
   return index_dist;
}

// --- no. of A/B/P/Q operators ---
inline int oper_num_opA(const int cindex1_size, const bool& ifkr){
   int k = ifkr? cindex1_size : cindex1_size/2;
   int num = ifkr? k*k : k*(2*k-1);
   return num;
}
inline int oper_num_opB(const int cindex1_size, const bool& ifkr){
   int k = ifkr? cindex1_size : cindex1_size/2;
   int num = ifkr? k*(k+1) : k*(2*k+1);
   return num;
}
inline int oper_num_opP(const int krest_size, const bool& ifkr){
   int k = krest_size;
   int num = ifkr? k*k : k*(2*k-1);
   return num;
}
inline int oper_num_opQ(const int krest_size, const bool& ifkr){
   int k = krest_size;
   int num = ifkr? k*(k+1) : k*(2*k+1);
   return num;
}

// --- generate indices for A/B operators from cindex --- 
inline std::vector<int> oper_index_opA(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> aindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
	 if(p1 < q1){ 
            aindex.push_back( oper_pack(p1,q1) );
            if(ifkr) aindex.push_back( oper_pack(p1,q1+1) );
	 }else if(p1 == q1){
	    if(ifkr) aindex.push_back( oper_pack(p1,p1+1) );
	 }
      }
   }
   assert(aindex.size() == oper_num_opA(cindex1.size(),ifkr));
   return aindex;
}
inline std::vector<int> oper_index_opA_dist(const std::vector<int>& cindex1, const bool& ifkr,
					    const int size, const int rank){
   std::vector<int> aindex = oper_index_opA(cindex1, ifkr);
   return distribute2(ifkr, size, aindex, rank);
}

inline std::vector<int> oper_index_opB(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> bindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 <= q1){
            bindex.push_back( oper_pack(p1,q1) );
	    if(ifkr) bindex.push_back( oper_pack(p1,q1+1) );
	 }
      }
   }
   assert(bindex.size() == oper_num_opB(cindex1.size(),ifkr));
   return bindex;
}
inline std::vector<int> oper_index_opB_dist(const std::vector<int>& cindex1, const bool& ifkr,
					    const int size, const int rank){
   std::vector<int> bindex = oper_index_opB(cindex1, ifkr);
   return distribute2(ifkr, size, bindex, rank);
}

inline std::vector<int> oper_index_opBdiag(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> bindex;
   for(int p1 : cindex1){
      bindex.push_back( oper_pack(p1,p1) );
   }
   assert(bindex.size() == cindex1.size());
   return bindex;
}
inline std::vector<int> oper_index_opBdiag_dist(const std::vector<int>& cindex1, const bool& ifkr,
					    const int size, const int rank){
   std::vector<int> bindex = oper_index_opBdiag(cindex1, ifkr);
   return distribute2(ifkr, size, bindex, rank);
}

// --- generate index for complementary operators: P,Q,S ---
// tricky part: determine the storage pattern for Ppq for p,q in krest
inline std::vector<int> oper_index_opP(const std::vector<int>& krest, const bool& ifkr){
   std::vector<int> index;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int kq : krest){
	 int qa = 2*kq, qb = qa+1;
	 if(kp < kq){
            index.push_back(oper_pack(pa,qa)); // Paa 
	    index.push_back(oper_pack(pa,qb)); // Pab
	    if(!ifkr){
	       // since if kp<kq, pb<qa and pb<qb hold
	       index.push_back(oper_pack(pb,qa));
	       index.push_back(oper_pack(pb,qb));
	    }
	 }else if(kp == kq){
            index.push_back(oper_pack(pa,pb)); // Pab 
	 }
      } // kq
   } // kp
   assert(index.size() == oper_num_opP(krest.size(),ifkr)); 
   return index;
}
inline std::vector<int> oper_index_opP_dist(const std::vector<int>& krest, const bool& ifkr,
					    const int size, const int rank){
   std::vector<int> pindex = oper_index_opP(krest, ifkr);
   return distribute2(ifkr, size, pindex, rank);
}

inline std::vector<int> oper_index_opQ(const std::vector<int>& krest, const bool& ifkr){
   std::vector<int> index;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int ks : krest){
	 int sa = 2*ks, sb = sa+1;
	 if(kp <= ks){ 
	    index.push_back(oper_pack(pa,sa));
	    index.push_back(oper_pack(pa,sb));
	    if(!ifkr){
	       // if kp=ks, QpApB is stored while QpBpA is redundant,
	       // because it can be related with QpApB using Hermiticity if bra=ket.
	       if(kp != ks) index.push_back(oper_pack(pb,sa));
	       index.push_back(oper_pack(pb,sb));
	    }
	 }
      } // ks
   } // kp
   assert(index.size() == oper_num_opQ(krest.size(),ifkr));
   return index;
}
inline std::vector<int> oper_index_opQ_dist(const std::vector<int>& krest, const bool& ifkr,
					    const int size, const int rank){
   std::vector<int> qindex = oper_index_opQ(krest, ifkr);
   return distribute2(ifkr, size, qindex, rank);
}

inline std::vector<int> oper_index_opS(const std::vector<int>& krest, const bool& ifkr){
   std::vector<int> index;
   for(int kp: krest){
      int pa = 2*kp, pb = pa+1;
      index.push_back(pa);
      if(!ifkr) index.push_back(pb);
   }
   return index;
}
inline std::vector<int> oper_index_opC(const std::vector<int>& ksupp, const bool& ifkr){
   return oper_index_opS(ksupp, ifkr);
}

// --- combination of two sets of indices ---
inline std::vector<int> oper_combine_cindex(const std::vector<int>& cindex1,
		                            const std::vector<int>& cindex2){
   std::vector<int> cindex;
   cindex.insert(cindex.end(), cindex1.begin(), cindex1.end());
   cindex.insert(cindex.end(), cindex2.begin(), cindex2.end());
   return cindex;
}

// --- formula for combinations --- 
inline std::vector<std::pair<int,int>> oper_combine_opC(const std::vector<int>& cindex1,
		      	     				const std::vector<int>& cindex2){
   std::vector<std::pair<int,int>> info;
   int iformula; 
   // 1. p1^+*I2
   iformula = 1;
   for(int p1 : cindex1){
      info.emplace_back(p1,iformula);
   }
   // 2. I1*p2^+ 
   iformula = 2;
   for(int p2 : cindex2){
      info.emplace_back(p2,iformula);
   }
   return info;
}

// tricky part: determine the storage pattern for Apq
// nkr: only store Apq where p<q: total number (2K)*(2K-1)/2 ~ O(2K^2)
//  kr: time-reversal symmetry adapted basis is used, Apq blocks:
//      pA+qA+ and pA+qB+: K*(K-1)/2+K*(K+1)/2=K^2 (reduction by half)
inline std::vector<std::pair<int,int>> oper_combine_opA(const std::vector<int>& cindex1,
		             			        const std::vector<int>& cindex2,
		             			        const bool& ifkr){
   std::vector<std::pair<int,int>> info;
   int iformula; 
   // 1. p1<q1: A[p1,q1] = p1^+q1^+ * I2
   iformula = 1;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 < q1){
            info.emplace_back(oper_pack(p1,q1),iformula);
            if(ifkr) info.emplace_back(oper_pack(p1,q1+1),iformula);
	 }else if(p1 == q1){                                      
	    if(ifkr) info.emplace_back(oper_pack(p1,p1+1),iformula);
	 }
      }
   }
   // 2. p2<q2: A[p2,q2] = I1 * p2^+q2^+ 
   iformula = 2;
   for(int p2 : cindex2){
      for(int q2 : cindex2){
         if(p2 < q2){
            info.emplace_back(oper_pack(p2,q2),iformula);
	    if(ifkr) info.emplace_back(oper_pack(p2,q2+1),iformula);
	 }else if(p2 == q2){                            
	    if(ifkr) info.emplace_back(oper_pack(p2,p2+1),iformula);
	 }
      }
   }
   // 3. p1<q2: A[p1,q2] = p1^+ * q2^+ 
   // 4. p1>q2: A[q2,p1] = q2^+ * p1^+ = -p1^+ * q2^+
   for(int p1 : cindex1){
      for(int q2 : cindex2){
         int index = (p1<q2)? oper_pack(p1,q2) : oper_pack(q2,p1);
	 iformula = (p1<q2)? 3 : 4;
	 info.emplace_back(index,iformula);
         if(ifkr){ // Opposite-spin part:
	    index = (p1<q2)? oper_pack(p1,q2+1) : oper_pack(q2,p1+1);
	    info.emplace_back(index,iformula);
	 }
      }
   }
   int kc = cindex1.size()+cindex2.size();
   assert(info.size() == oper_num_opA(kc,ifkr));
   return info;
}

// tricky part: determine the storage pattern for Bps
// nkr: only store Bps (p<=s): (2K)*(2K+1)/2 ~ O(2K^2)
//  kr: If time-reversal symmetry adapted basis is used, Bps blocks:
//      pA+sA and pA+sB: K*(K+1)/2+K*(K+1)/2=K(K+1) (reduction by half)
inline std::vector<std::pair<int,int>> oper_combine_opB(const std::vector<int>& cindex1,
		             			        const std::vector<int>& cindex2,
		             			        const bool& ifkr){
   std::vector<std::pair<int,int>> info;
   int iformula; 
   // 1. p1<q1: B[p1,q1] = p1^+q1 * I2
   iformula = 1;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 <= q1){
            info.emplace_back(oper_pack(p1,q1),iformula);
	    if(ifkr) info.emplace_back(oper_pack(p1,q1+1),iformula);
	 }
      }
   }
   // 2. p2<=q2: B[p2,q2] = I1 * p2^+q2
   iformula = 2;
   for(int p2 : cindex2){
      for(int q2 : cindex2){
         if(p2 <= q2){
            info.emplace_back(oper_pack(p2,q2),iformula);
	    if(ifkr) info.emplace_back(oper_pack(p2,q2+1),iformula);
	 }
      }
   }
   // 3. p1<q2: B[p1,q2] = p1^+ * q2
   // 4. p1>q2: B[q2,p1] = q2 * p1^+ = -p1^+ * q2
   for(int p1 : cindex1){
      for(int q2 : cindex2){
         int index = (p1<q2)? oper_pack(p1,q2) : oper_pack(q2,p1);
	 iformula = (p1<q2)? 3 : 4;
	 info.emplace_back(index,iformula);
         if(ifkr){ // Opposite-spin part:
	    index = (p1<q2)? oper_pack(p1,q2+1) : oper_pack(q2,p1+1);
	    info.emplace_back(index,iformula);
	 }
      }
   }
   int kc = cindex1.size()+cindex2.size();
   assert(info.size() == oper_num_opB(kc,ifkr));
   return info;
}

} // ctns

#endif
