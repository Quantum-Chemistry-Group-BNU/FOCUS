#ifndef CTNS_OPER_DOT_H
#define CTNS_OPER_DOT_H

#include "ctns_phys.h"
#include "ctns_oper_util.h"

namespace ctns{

// We use the convention that p1+p2+q2q1 where p1<p2 and q2>q1, i.e., 
// The index in the middle is larger than that close to the boundary.
// This is different from the ordering used in onstate.h

// pA^+
template <typename Tm>
void oper_dot_C(const int k0, oper_dict<Tm>& qops){
   auto qs_phys = get_qsym_space_phys<Tm>();
   const bool Htype = tools::is_complex<Tm>();
   // c[0]
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [1. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,3) = 1;
   mat(2,0) = 1;
   auto sym_op = Htype? qsym(1,0) : qsym(1,1);
   qtensor2<Tm> qt2(sym_op, qs_phys, qs_phys);
   qt2.from_matrix(mat);
   qops['C'][k0] = qt2;
/*
   qops['C'][ka].to_matrix().print("c0");
   auto a0 = qops['C'][ka].H();
   a0.to_matrix().print("a0");

   auto c1 = qops['C'][ka].K(1);
   c1.to_matrix().print("c1");
   auto a1 = c1.H();
   a1.to_matrix().print("a1");
   exit(1);
*/
}

// A[pA,pB] = pA^+pB^+
template <typename Tm>
void oper_dot_A(const int k0, oper_dict<Tm>& qops){
   auto qs_phys = get_qsym_space_phys<Tm>();
   // c[0].dot(c[1])
   // [[0. 0. 0. 0.]
   //  [1. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,0) = 1;
   auto sym_op = qsym(2,0);
   qtensor2<Tm> qt2(sym_op, qs_phys, qs_phys);
   qt2.from_matrix(mat);
   qops['A'][oper_pack(1,k0,k0)] = qt2;
/* 
   qt2.print("qt2",2);
   qt2.to_matrix().print("A0");

   auto A1 = qops['A'][oper_pack(ka,kb)].K(1);
   A1.to_matrix().print("A1");
   exit(1);
*/
}

// Ppq = <pq||sr> aras [p<q,r>s] = <pq||sr> A[sr]^+
// P[pA,qA] = <pA,qA||kA,kB> A[kA,kB]^+ (zero in NR)
// P[pA,qB] = <pA,qB||kA,kB> A[kA,kB]^+
template <typename Tm>
void oper_dot_P(const int k0,
		const integral::two_body<Tm>& int2e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   // P[pA,qA] (p<q) and P[pA,qB] (p<=q)
   auto qt2ab = qops['A'][oper_pack(1,k0,k0)].H();
   int ka = 2*k0, kb = ka+1;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int kq : krest){
	 int qa = 2*kq, qb = qa+1;
	 if(kp < kq){
	    qops['P'][oper_pack(0,kp,kq)] = int2e.get(pa,qa,ka,kb)*qt2ab;
	 }
	 if(kp <= kq){
	    qops['P'][oper_pack(1,kp,kq)] = int2e.get(pa,qb,ka,kb)*qt2ab;
	 }
      }
   }
}

// B[pA,pA] = pA^+pA, B[pA,pB] = pA^+pB
template <typename Tm>
void oper_dot_B(const int k0, oper_dict<Tm>& qops){
   auto qs_phys = get_qsym_space_phys<Tm>();
   const bool Htype = tools::is_complex<Tm>();
   // c[0].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mataa(4,4);
   mataa(1,1) = 1;
   mataa(2,2) = 1;
   qtensor2<Tm> qt2aa(qsym(0,0), qs_phys, qs_phys);
   qt2aa.from_matrix(mataa);
   qops['B'][oper_pack(0,k0,k0)] = qt2aa;
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> matab(4,4);
   matab(2,3) = 1;
   auto sym_op = Htype? qsym(0,0) : qsym(0,1);
   qtensor2<Tm> qt2ab(sym_op, qs_phys, qs_phys);
   qt2ab.from_matrix(matab);
   qops['B'][oper_pack(1,k0,k0)] = qt2ab;
/*
   qt2aa.print("qt2aa",2);
   qt2aa.to_matrix().print("Baa");
   auto Bbb = qt2aa.K(0);
   Bbb.to_matrix().print("Bbb");

   qt2ab.print("qt2ab",2);
   qt2ab.to_matrix().print("Bab");
   auto Bba = qt2ab.K(1);
   Bba.to_matrix().print("Bba");
   exit(1);
*/
}

// Qps = <pq||sr> aq^+ar
// Q[pA,sA] = <pA,q||sA,r> B[q,r]
// 	    = <pA,kA||sA,kA> B[kA,kA] + <pA,kB||sA,kB> B[kB,kB]
//	    + <pA,kA||sA,kB> B[kA,kB] + <pA,kB||sA,kA> B[kB,kA] (zero in NR)
// Q[pA,sB] = <pA,q||sB,r> B[q,r]
// 	    = <pA,kA||sB,kA> B[kA,kA] + <pA,kB||sB,kB> B[kB,kB] 
//	    + <pA,kA||sB,kB> B[kA,kB] + <pA,kB||sB,kA> B[kB,kA] (nonzero) 
template <typename Tm>
void oper_dot_Q(const int k0,
		const integral::two_body<Tm>& int2e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   auto& qt2aa = qops['B'][oper_pack(0,k0,k0)];
   auto& qt2ab = qops['B'][oper_pack(1,k0,k0)];
   auto qt2bb = qt2aa.K(0);
   auto qt2ba = qt2ab.K(1); 
   // Q[pA,sA] (p<=s) and Q[pA,sB] (p<=s)
   int ka = 2*k0, kb = ka+1;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      for(int ks : krest){
	 int sa = 2*ks, sb = sa+1;
	 if(kp <= ks){
	    qops['Q'][oper_pack(0,kp,ks)] = int2e.get(pa,ka,sa,ka)*qt2aa
		    		          + int2e.get(pa,kb,sa,kb)*qt2bb
				   	  + int2e.get(pa,ka,sa,kb)*qt2ab
					  + int2e.get(pa,kb,sa,ka)*qt2ba;
	    qops['Q'][oper_pack(1,kp,ks)] = int2e.get(pa,ka,sb,ka)*qt2aa
	    			          + int2e.get(pa,kb,sb,kb)*qt2bb
					  + int2e.get(pa,ka,sb,kb)*qt2ab
					  + int2e.get(pa,kb,sb,ka)*qt2ba;
	 }
      }
   }
}

// build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Tm>
void oper_dot_S(const int k0,
		const integral::two_body<Tm>& int2e,
		const integral::one_body<Tm>& int1e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   auto qs_phys = get_qsym_space_phys<Tm>();
   const bool Htype = tools::is_complex<Tm>();
   // c[0].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(2,1) = 1;
   auto sym_op = Htype? qsym(-1,0) : qsym(-1,1);
   qtensor2<Tm> qt2aba(sym_op, qs_phys, qs_phys); // ka^+ kb ka
   qt2aba.from_matrix(mat);
   // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   int ka = 2*k0, kb = ka+1;
   for(int kp : krest){
      int pa = 2*kp, pb = pa+1;
      auto Spa = 0.5*int1e.get(pa,ka)*qops['C'][k0].H()
	       - int2e.get(pa,kb,ka,kb)*qt2aba.K(0);
      if(Htype){
	 Spa += 0.5*int1e.get(pa,kb)*qops['C'][k0].K(1).H()
	      + int2e.get(pa,ka,ka,kb)*qt2aba;		 
      }
      qops['S'][kp] = Spa;
/*
      Spa.to_matrix().print("Spa");
*/
   } // p
}

// build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
template <typename Tm>
void oper_dot_H(const int k0,
		const integral::two_body<Tm>& int2e,
		const integral::one_body<Tm>& int1e,
		oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
   auto qs_phys = get_qsym_space_phys<Tm>();
   const bool Htype = tools::is_complex<Tm>();
   // 0110 (*<01||01>)
   // c[0].dot(c[1].dot(a[1].dot(a[0])))
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   linalg::matrix<Tm> mat(4,4);
   mat(1,1) = 1;
   qtensor2<Tm> qt2abba(qsym(0,0), qs_phys, qs_phys);
   qt2abba.from_matrix(mat);
   // <ka,kb||ka,kb> ka^+ kb^+ kb ka
   auto qt2 = int2e.get(ka,kb,ka,kb)*qt2abba;
   // h[ka,ka] ka^+ka + h[kb,kb] kb^+kb
   const auto& qop = qops['B'][oper_pack(0,k0,k0)];
   qt2 += int1e.get(ka,ka)*qop + int1e.get(kb,kb)*qop.K(0);
   if(Htype){
      // h[ka,kb] ka^+kb + h[kb,ka] kb^+ka
      const auto& qop = qops['B'][oper_pack(1,k0,k0)];
      qt2 += int1e.get(ka,kb)*qop + int1e.get(kb,ka)*qop.K(1); 
   }
   qops['H'][0] = qt2;
}

} // ctns

#endif
