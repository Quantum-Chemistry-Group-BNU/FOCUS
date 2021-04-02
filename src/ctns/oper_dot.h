#ifndef OPER_DOT_H
#define OPER_DOT_H

#include "ctns_phys.h"

namespace ctns{

/*
 dot operators: C/A/B/P/Q/S/H

 The local basis is {|0>,|2>,|a>,|b>} in consistent with ctns_phys.h

 We use the convention that p1+*p2+*q2*q1 where p1<p2 and q2>q1, i.e., 
 The index in the middle is larger than that close to the boundary.
 This is different from the ordering used in onstate.h
*/

// kA^+
template <typename Tm>
void oper_dot_C(const int isym, const bool ifkr, const int k0, 
		oper_dict<Tm>& qops){
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
   // c[1] = kB^+ 
   if(not ifkr){
      qops['C'][kb] = qt2.K(1);
   }
}

// A[kA,kB] = kA^+kB^+
template <typename Tm>
void oper_dot_A(const int isym, const bool ifkr, const int k0, 
		oper_dict<Tm>& qops){
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
}

// B[kA,kA] = kA^+kA, B[kA,kB] = kA^+kB
// B[kB,kA] = kB^+kA = B[kA,kB].K(1)
// B[kB,kB] = kB^+kB = B[kA,kA].K(2)
template <typename Tm>
void oper_dot_B(const int isym, const bool ifkr, const int k0, 
		oper_dict<Tm>& qops){
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
   qtensor2<Tm> qt2aa(qsym(0,0), qphys, qphys);
   qt2aa.from_matrix(mataa);
   qops['B'][oper_pack(ka,ka)] = qt2aa;
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> matab(4,4);
   matab(2,3) = 1;
   auto sym_op = (isym == 1)? qsym(0,0) : qsym(0,1);
   qtensor2<Tm> qt2ab(sym_op, qphys, qphys);
   qt2ab.from_matrix(matab);
   qops['B'][oper_pack(ka,kb)] = qt2ab;
   // Bba and Bbb
   if(not ifkr){
      qops['B'][oper_pack(kb,ka)] = qt2ab.K(1);
      qops['B'][oper_pack(kb,kb)] = qt2aa.K(2);
   }
}

// build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Tm>
void oper_dot_S(const int isym, const bool ifkr, const int k0,
		const integral::two_body<Tm>& int2e,
		const integral::one_body<Tm>& int1e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
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
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      qops['S'][pa] = 0.5*int1e.get(pa,ka)*qops['C'][ka].H()
	            + 0.5*int1e.get(pa,kb)*qops['C'][ka].K(1).H()
                    + int2e.get(pa,ka,ka,kb)*qt2aba       // a^+ba  		 
                    - int2e.get(pa,kb,ka,kb)*qt2aba.K(2); // b^+ba = (Kb^+ba)^* = a^+ab* = -a^+ba*
      // also save S_{pB}^C
      if(not ifkr){
         qops['S'][pb] = qops['S'][pa].K(1);
      }
   } // kp
}

// build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
template <typename Tm>
void oper_dot_H(const int isym, const bool ifkr, const int k0,
		const integral::two_body<Tm>& int2e,
		const integral::one_body<Tm>& int1e,
		oper_dict<Tm>& qops){
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
   // it is fine to use K(), because local space is time-reversal adapted!
   qops['H'][0] = int2e.get(ka,kb,ka,kb)*qt2abba
                + int1e.get(ka,ka)*qBaa + int1e.get(kb,kb)*qBaa.K(2);
   		+ int1e.get(ka,kb)*qBab + int1e.get(kb,ka)*qBab.K(1);
}

// Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
// P[pA,qA] = <pA,qA||kA,kB> A[kA,kB]^+ (zero in NR)
// P[pA,qB] = <pA,qB||kA,kB> A[kA,kB]^+
template <typename Tm>
void oper_dot_P(const int isym, const bool ifkr, const int k0,
		const integral::two_body<Tm>& int2e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
   auto qt2ab = qops['A'][oper_pack(ka,kb)].H();
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int kq : krest){
	 int qa = 2*kq, qb = qa+1;
	 // storage scheme for Ppq: according to spatial orbital
	 if(kp < kq){
	    qops['P'][oper_pack(pa,qa)] = int2e.get(pa,qa,ka,kb)*qt2ab;
	    qops['P'][oper_pack(pa,qb)] = int2e.get(pa,qb,ka,kb)*qt2ab;
	    if(not ifkr){
	       qops['P'][oper_pack(pb,qa)] = qops['P'][oper_pack(pa,qb)].K(1);
	       qops['P'][oper_pack(pb,qb)] = qops['P'][oper_pack(pa,qa)].K(2);
	    }
	 }else if(kp == kq){
	    qops['P'][oper_pack(pa,pb)] = int2e.get(pa,pb,ka,kb)*qt2ab;
	 }
      } // kq
   } // kp
}

// Qps = <pq||sr> aq^+ar
// Q[pA,sA] = <pA,q||sA,r> B[q,r]
// 	    = <pA,kA||sA,kA> B[kA,kA] + <pA,kB||sA,kB> B[kB,kB]
//	    + <pA,kA||sA,kB> B[kA,kB] + <pA,kB||sA,kA> B[kB,kA] (zero in NR)
// Q[pA,sB] = <pA,q||sB,r> B[q,r]
// 	    = <pA,kA||sB,kA> B[kA,kA] + <pA,kB||sB,kB> B[kB,kB] 
//	    + <pA,kA||sB,kB> B[kA,kB] + <pA,kB||sB,kA> B[kB,kA] (nonzero) 
template <typename Tm>
void oper_dot_Q(const int isym, const bool ifkr, const int k0,
		const integral::two_body<Tm>& int2e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
   const auto& qt2aa = qops['B'][oper_pack(ka,ka)];
   const auto& qt2ab = qops['B'][oper_pack(ka,kb)];
   auto qt2ba = qt2ab.K(1); 
   auto qt2bb = qt2aa.K(2);
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int ks : krest){
	 int sa = 2*ks, sb = sa+1;
	 // storage scheme for Qps: according to spatial orbital
	 if(kp <= ks){
            qops['Q'][oper_pack(pa,sa)] = int2e.get(pa,ka,sa,ka)*qt2aa // <p ka||s ka> ka^+ka 
               		                + int2e.get(pa,kb,sa,kb)*qt2bb
      		                        + int2e.get(pa,ka,sa,kb)*qt2ab
      		                        + int2e.get(pa,kb,sa,ka)*qt2ba;
	    qops['Q'][oper_pack(pa,sb)] = int2e.get(pa,ka,sb,ka)*qt2aa
	    			        + int2e.get(pa,kb,sb,kb)*qt2bb
					+ int2e.get(pa,ka,sb,kb)*qt2ab
					+ int2e.get(pa,kb,sb,ka)*qt2ba;
	    if(not ifkr){
	       qops['Q'][oper_pack(pb,sa)] = qops['Q'][oper_pack(pa,sb)].K(1);
	       qops['Q'][oper_pack(pb,sb)] = qops['Q'][oper_pack(pa,sa)].K(2);
	    }
	 }
      } // ks
   } // kp
}

} // ctns

#endif
