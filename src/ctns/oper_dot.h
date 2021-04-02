#ifndef OPER_DOT_H
#define OPER_DOT_H

#include "ctns_phys.h"

namespace ctns{

const bool debug_dot = false;
extern const bool debug_dot;

/*
 dot operators: C/A/B/P/Q/S/H

 The local basis is {|0>,|2>,|a>,|b>} in consistent with ctns_phys.h

 We use the convention that p1+*p2+*q2*q1 where p1<p2 and q2>q1, i.e., 
 The index in the middle is larger than that close to the boundary.
 This is different from the ordering used in onstate.h
*/

// kA^+
template <typename Tm>
void oper_dot_opC(const int isym, const bool ifkr, const int k0, 
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opC" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qphys = get_qbond_phys(isym);
   // c[0] = kA^+
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [1. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,3) = 1;
   mat(2,0) = 1;
   auto sym_op = (isym == 1)? qsym(1,0) : qsym(1,1);
   qtensor2<Tm> qt2(sym_op, qphys, qphys);
   qt2.from_matrix(mat);
   qops['C'][ka] = qt2;
   if(debug_dot) qops['C'][ka].to_matrix().print("c0+");
   // c[1] = kB^+ 
   if(not ifkr){
      // also store c[1] = kB^+
      // [[ 0.  0.  0.  0.]
      //  [ 0.  0. -1.  0.]
      //  [ 0.  0.  0.  0.]
      //  [ 1.  0.  0.  0.]]
      linalg::matrix<Tm> mat(4,4);
      mat(1,2) = -1;
      mat(3,0) = 1;
      auto sym_op = (isym == 1)? qsym(1,0) : qsym(1,-1);
      qtensor2<Tm> qt2(sym_op, qphys, qphys);
      qt2.from_matrix(mat);
      qops['C'][kb] = qt2;
      if(debug_dot) qops['C'][kb].to_matrix().print("c1+");
   }
}

// A[kA,kB] = kA^+kB^+
template <typename Tm>
void oper_dot_opA(const int isym, const bool ifkr, const int k0, 
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opA" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qphys = get_qbond_phys(isym);
   // c[0].dot(c[1])
   // [[0. 0. 0. 0.]
   //  [1. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,0) = 1;
   auto sym_op = qsym(2,0);
   qtensor2<Tm> qt2(sym_op, qphys, qphys);
   qt2.from_matrix(mat);
   qops['A'][oper_pack(ka,kb)] = qt2;
   if(debug_dot) qops['A'][oper_pack(ka,kb)].to_matrix().print("c0^+c1^+");
}

// B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
// B[kB,kA] = kB^+kA = B[kA,kB].K(1)
// B[kB,kB] = kB^+kB = B[kA,kA].K(2)
template <typename Tm>
void oper_dot_opB(const int isym, const bool ifkr, const int k0, 
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opB" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qphys = get_qbond_phys(isym);
   // c[0].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mataa(4,4);
   mataa(1,1) = 1;
   mataa(2,2) = 1;
   qtensor2<Tm> qt2aa(qsym(0,0), qphys, qphys); // a^+a
   qt2aa.from_matrix(mataa);
   qops['B'][oper_pack(ka,ka)] = qt2aa;
   if(debug_dot) qops['B'][oper_pack(ka,ka)].to_matrix().print("c0^+c0");
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> matab(4,4);
   matab(2,3) = 1;
   auto sym_op = (isym == 1)? qsym(0,0) : qsym(0,2); // a^+b
   qtensor2<Tm> qt2ab(sym_op, qphys, qphys);
   qt2ab.from_matrix(matab);
   qops['B'][oper_pack(ka,kb)] = qt2ab;
   if(debug_dot) qops['B'][oper_pack(ka,kb)].to_matrix().print("c0^+c1");
   // Bba and Bbb
   if(not ifkr){
      // c[1].dot(a[0])
      // [[0. 0. 0. 0.]
      //  [0. 0. 0. 0.]
      //  [0. 0. 0. 0.]
      //  [0. 0. 1. 0.]]
      linalg::matrix<Tm> matba(4,4);
      matba(3,2) = 1;
      auto sym_op = (isym == 1)? qsym(0,0) : qsym(0,-2); // b^+a
      qtensor2<Tm> qt2ba(sym_op, qphys, qphys);
      qt2ba.from_matrix(matba);
      qops['B'][oper_pack(kb,ka)] = qt2ba;
      if(debug_dot) qops['B'][oper_pack(kb,ka)].to_matrix().print("c1^+c0");
      // c[1].dot(a[1])
      // [[0. 0. 0. 0.]
      //  [0. 1. 0. 0.]
      //  [0. 0. 0. 0.]
      //  [0. 0. 0. 1.]]
      linalg::matrix<Tm> matbb(4,4);
      matbb(1,1) = 1;
      matbb(3,3) = 1;
      qtensor2<Tm> qt2bb(qsym(0,0), qphys, qphys);
      qt2bb.from_matrix(matbb);
      qops['B'][oper_pack(kb,kb)] = qt2bb;
      if(debug_dot) qops['B'][oper_pack(kb,kb)].to_matrix().print("c1^+c1");
   }
}

// build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Tm>
void oper_dot_opS(const int isym, const bool ifkr, const int k0,
		  const integral::two_body<Tm>& int2e,
		  const integral::one_body<Tm>& int1e,
		  const std::vector<int>& krest,
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opS" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qphys = get_qbond_phys(isym);
   // c[0].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(2,1) = 1;
   auto sym_op = (isym == 1)? qsym(-1,0) : qsym(-1,1);
   qtensor2<Tm> qt2aba(sym_op, qphys, qphys); // ka^+ kb ka
   qt2aba.from_matrix(mat);
   // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   if(ifkr){
      for(int kp : krest){
         int pa = 2*kp, pb = pa+1;
         qops['S'][pa] = 0.5*int1e.get(pa,ka)*qops['C'][ka].H()
                       + 0.5*int1e.get(pa,kb)*qops['C'][ka].K(1).H()
                       + int2e.get(pa,ka,ka,kb)*qt2aba       // a^+ba  		 
                       - int2e.get(pa,kb,ka,kb)*qt2aba.K(2); // b^+ba = (Kb^+ba)^* = a^+ab* = -a^+ba*
      }
   }else{
      // c[1].dot(a[1].dot(a[0]))
      // [[0. 0. 0. 0.]
      //  [0. 0. 0. 0.]
      //  [0. 0. 0. 0.]
      //  [0. 1. 0. 0.]]
      linalg::matrix<Tm> mat(4,4);
      mat(3,1) = 1;
      auto sym_op = (isym == 1)? qsym(-1,0) : qsym(-1,-1);
      qtensor2<Tm> qt2bba(sym_op, qphys, qphys);
      qt2bba.from_matrix(mat);
      const auto akA = qops['C'][ka].H();
      const auto akB = qops['C'][kb].H();
      if(isym == 1){
         for(int kp : krest){
            int pa = 2*kp, pb = pa+1;
            // S_{pA} += <pAkA||kAkB> kA^+kBkA + <pAkB||kAkB> kB^+kBkA 
            qops['S'][pa] = 0.5*int1e.get(pa,ka)*akA
           	          + 0.5*int1e.get(pa,kb)*akB       // zero in NR
           	          + int2e.get(pa,ka,ka,kb)*qt2aba  // zero in NR
           	          + int2e.get(pa,kb,ka,kb)*qt2bba;
            // S_{pB} += <pBkA||kAkB> kA^+kBkA + <pBkB||kAkB> kB^+kBkA 
            qops['S'][pb] = 0.5*int1e.get(pb,ka)*akA       // zero in NR
           	          + 0.5*int1e.get(pb,kb)*akB
           	          + int2e.get(pb,ka,ka,kb)*qt2aba 
           	          + int2e.get(pb,kb,ka,kb)*qt2bba; // zero in NR
         }
      }else if(isym == 2){
         for(int kp : krest){
            int pa = 2*kp, pb = pa+1;
            qops['S'][pa] = 0.5*int1e.get(pa,ka)*akA
           	          + int2e.get(pa,kb,ka,kb)*qt2bba;
            qops['S'][pb] = 0.5*int1e.get(pb,kb)*akB
           	          + int2e.get(pb,ka,ka,kb)*qt2aba; 
         }
      } // isym
   } // kp
}

// build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
template <typename Tm>
void oper_dot_opH(const int isym, const bool ifkr, const int k0,
		  const integral::two_body<Tm>& int2e,
		  const integral::one_body<Tm>& int1e,
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opH" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qphys = get_qbond_phys(isym);
   // 0110 (*<01||01>)
   // c[0].dot(c[1].dot(a[1].dot(a[0])))
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,1) = 1;
   qtensor2<Tm> qt2abba(qsym(0,0), qphys, qphys);
   qt2abba.from_matrix(mat);
   const auto& qBaa = qops['B'][oper_pack(ka,ka)];
   const auto& qBab = qops['B'][oper_pack(ka,kb)];
   if(ifkr){
      qops['H'][0] = int2e.get(ka,kb,ka,kb)*qt2abba
                   + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBaa.K(2);
                   + int1e.get(ka,kb)*qBab + int1e.get(kb,ka)*qBab.K(1);
   }else{
      const auto& qBba = qops['B'][oper_pack(kb,ka)];
      const auto& qBbb = qops['B'][oper_pack(kb,kb)];
      if(isym == 1){
         qops['H'][0] = int2e.get(ka,kb,ka,kb)*qt2abba
                      + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBbb
                      + int1e.get(ka,kb)*qBab + int1e.get(kb,ka)*qBba;
      }else if(isym == 2){
         qops['H'][0] = int2e.get(ka,kb,ka,kb)*qt2abba
                      + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBbb;
      }
   }
}

// Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
template <typename Tm>
void oper_dot_opP(const int isym, const bool ifkr, const int k0,
		  const integral::two_body<Tm>& int2e,
		  const std::vector<int>& krest,
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opP" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   auto qt2ab = qops['A'][oper_pack(ka,kb)].H();
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int kq : krest){
	 int qa = 2*kq, qb = qa+1;
	 // storage scheme for Ppq: according to spatial orbital
	 if(kp < kq){
            // P[pA,qA] = <pA,qA||kA,kB> A[kA,kB]^+ (zero in NR)
	    qops['P'][oper_pack(pa,qa)] = int2e.get(pa,qa,ka,kb)*qt2ab; 
            // P[pA,qB] = <pA,qB||kA,kB> A[kA,kB]^+
	    qops['P'][oper_pack(pa,qb)] = int2e.get(pa,qb,ka,kb)*qt2ab;
	    if(not ifkr){
               // P[pB,qA] = <pB,qA||kA,kB> A[kA,kB]^+ 
	       qops['P'][oper_pack(pb,qa)] = int2e.get(pb,qa,ka,kb)*qt2ab; 
               // P[pB,qB] = <pB,qB||kA,kB> A[kA,kB]^+ (zero in NR)
	       qops['P'][oper_pack(pb,qb)] = int2e.get(pb,qb,ka,kb)*qt2ab;
	    }
	 }else if(kp == kq){
	    qops['P'][oper_pack(pa,pb)] = int2e.get(pa,pb,ka,kb)*qt2ab;
	 }
      } // kq
   } // kp
}

// Qps = <pq||sr> aq^+ar
template <typename Tm>
void oper_dot_opQ(const int isym, const bool ifkr, const int k0,
		  const integral::two_body<Tm>& int2e,
		  const std::vector<int>& krest,
		  oper_dict<Tm>& qops){
   if(debug_dot) std::cout << "ctns::oper_dot_opQ" << std::endl; 
   int ka = 2*k0, kb = ka+1;
   const auto& qt2aa = qops['B'][oper_pack(ka,ka)];
   const auto& qt2ab = qops['B'][oper_pack(ka,kb)];
   qtensor2<Tm> qt2ba, qt2bb;
   if(ifkr){
      qt2ba = qt2ab.K(1); 
      qt2bb = qt2aa.K(2);
      for(int kp : krest){
         int pa = 2*kp, pb = pa+1;
         for(int ks : krest){
            int sa = 2*ks, sb = sa+1;
            // storage scheme for Qps: according to spatial orbital
            if(kp <= ks){
               // Q[pA,sA] = <pA,q||sA,r> B[q,r]
               // 	    = <pA,kA||sA,kA> B[kA,kA] + <pA,kB||sA,kB> B[kB,kB]
               //	    + <pA,kA||sA,kB> B[kA,kB] + <pA,kB||sA,kA> B[kB,kA] (zero in NR)
               qops['Q'][oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
                  		           + int2e.get(pa,kb,sa,kb)*qt2bb
         		                   + int2e.get(pa,ka,sa,kb)*qt2ab
         		                   + int2e.get(pa,kb,sa,ka)*qt2ba;
               // Q[pA,sB] = <pA,q||sB,r> B[q,r]
               // 	    = <pA,kA||sB,kA> B[kA,kA] + <pA,kB||sB,kB> B[kB,kB] (zero in NR) 
               //	    + <pA,kA||sB,kB> B[kA,kB] + <pA,kB||sB,kA> B[kB,kA] 
               qops['Q'][oper_pack(pa,sb)] = int2e.get(pa,ka,sb,ka)*qt2aa
               			           + int2e.get(pa,kb,sb,kb)*qt2bb
           				   + int2e.get(pa,ka,sb,kb)*qt2ab
           				   + int2e.get(pa,kb,sb,ka)*qt2ba;
            }
         } // ks
      } // kp
   }else{
      qt2ba = qops['B'][oper_pack(kb,ka)];
      qt2bb = qops['B'][oper_pack(kb,kb)];
      if(isym == 1){
         for(int kp : krest){
            int pa = 2*kp, pb = pa+1;
            for(int ks : krest){
               int sa = 2*ks, sb = sa+1;
               // storage scheme for Qps: according to spatial orbital
               if(kp <= ks){
                  qops['Q'][oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
                  		              + int2e.get(pa,kb,sa,kb)*qt2bb
         		                      + int2e.get(pa,ka,sa,kb)*qt2ab
         		                      + int2e.get(pa,kb,sa,ka)*qt2ba;
                  qops['Q'][oper_pack(pa,sb)] = int2e.get(pa,ka,sb,ka)*qt2aa
               			              + int2e.get(pa,kb,sb,kb)*qt2bb
           				      + int2e.get(pa,ka,sb,kb)*qt2ab
           				      + int2e.get(pa,kb,sb,ka)*qt2ba;
                  qops['Q'][oper_pack(pb,sa)] = int2e.get(pb,ka,sa,ka)*qt2aa 
                     		              + int2e.get(pb,kb,sa,kb)*qt2bb
         	                              + int2e.get(pb,ka,sa,kb)*qt2ab
         	                              + int2e.get(pb,kb,sa,ka)*qt2ba;
                  qops['Q'][oper_pack(pb,sb)] = int2e.get(pb,ka,sb,ka)*qt2aa
                  			      + int2e.get(pb,kb,sb,kb)*qt2bb
                      			      + int2e.get(pb,ka,sb,kb)*qt2ab
                      			      + int2e.get(pb,kb,sb,ka)*qt2ba;
	       }
            } // ks
         } // kp
      }else if(isym == 2){
         for(int kp : krest){
            int pa = 2*kp, pb = pa+1;
            for(int ks : krest){
               int sa = 2*ks, sb = sa+1;
               // storage scheme for Qps: according to spatial orbital
               if(kp <= ks){
	          // NOTE: zero terms need to be removed, otherwise qsym does not match!
                  qops['Q'][oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa 
                  		              + int2e.get(pa,kb,sa,kb)*qt2bb;
                  qops['Q'][oper_pack(pa,sb)] = int2e.get(pa,kb,sb,ka)*qt2ba;
                  qops['Q'][oper_pack(pb,sa)] = int2e.get(pb,ka,sa,kb)*qt2ab;
                  qops['Q'][oper_pack(pb,sb)] = int2e.get(pb,ka,sb,ka)*qt2aa
                  			      + int2e.get(pb,kb,sb,kb)*qt2bb;
	       }
            } // ks
         } // kp
      } // isym
   } // ifkr
}

} // ctns

#endif
