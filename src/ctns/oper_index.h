#ifndef OPER_SETTINGS_H
#define OPER_SETTINGS_H

namespace ctns{

// --- debug options ---
const bool debug_oper_dict = false;
extern const bool debug_oper_dict;

const bool debug_oper_io = false;
extern const bool debug_oper_io;

const bool debug_oper_para = false;
extern const bool debug_oper_para;

const bool debug_oper_dot = false;
extern const bool debug_oper_dot;

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

// --- distribution of (p,q) for MPI ---
inline int distribute2(const int index, const int size){
   auto pq = oper_unpack(index);
   int p = pq.first, q = pq.second;
   assert(p <= q);
   return (p == q)? p%size : (q*(q-1)/2+p)%size;
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

//--- generate index for complementary operators: P,Q,S ---
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
	    if(not ifkr){
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

inline std::vector<int> oper_index_opQ(const std::vector<int>& krest, const bool& ifkr){
   std::vector<int> index;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int ks : krest){
	 int sa = 2*ks, sb = sa+1;
	 if(kp <= ks){ 
	    index.push_back(oper_pack(pa,sa));
	    index.push_back(oper_pack(pa,sb));
	    if(not ifkr){
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

inline std::vector<int> oper_index_opS(const std::vector<int>& krest, const bool& ifkr){
   std::vector<int> index;
   for(int kp: krest){
      int pa = 2*kp, pb = pa+1;
      index.push_back(pa);
      if(not ifkr) index.push_back(pb);
   }
   return index;
}

} // ctns

#endif
