#ifndef CTNS_OPER_DOT_H
#define CTNS_OPER_DOT_H

#include "ctns_phys.h"
#include "ctns_oper_util.h"

namespace ctns{

// We use the convention that p1+p2+q2q1 where p1<p2 and q2>q1, i.e., 
// The index in the middle is larger than that close to the boundary.
// This is different from the ordering used in onstate.h

// <l'|ap^+|l>
template <typename Tm>
void oper_dot_C(const int k0, oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
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
   qops['C'][ka] = qt2;
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

// 01 (c0<c1)
template <typename Tm>
void oper_dot_A(const int k0, oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
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
   qops['A'][oper_pack(ka,kb)] = qt2;
/* 
   qt2.print("qt2",2);
   qt2.to_matrix().print("A0");

   auto A1 = qops['A'][oper_pack(ka,kb)].K(1);
   A1.to_matrix().print("A1");
   exit(1);
*/
}

// 00,01,(10),11
template <typename Tm>
void oper_dot_B(const int k0, oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
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
   qops['B'][oper_pack(ka,ka)] = qt2aa;
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
   qops['B'][oper_pack(ka,kb)] = qt2ab;
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
 
// build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
template <typename Tm>
void oper_dot_S(const int k0,
		const integral::two_body<Tm>& int2e,
		const integral::one_body<Tm>& int1e,
		const std::vector<int>& krest,
		oper_dict<Tm>& qops){
   int ka = 2*k0, kb = ka+1;
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
   for(int korb_p : krest){
      int pa = 2*korb_p, pb = pa+1;
      auto Spa = 0.5*int1e.get(pa,ka)*qops['C'][ka].H()
	       - int2e.get(pa,kb,ka,kb)*qt2aba.K(0);
      if(Htype){
	 Spa += 0.5*int1e.get(pa,kb)*qops['C'][ka].K(1).H()
	      + int2e.get(pa,ka,ka,kb)*qt2aba;		 
      }
      qops['S'][pa] = Spa;

      Spa.to_matrix().print("Spa");
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
   const auto& qop = qops['B'][oper_pack(ka,ka)];
   qt2 += int1e.get(ka,ka)*qop + int1e.get(kb,kb)*qop.K(0);
   if(Htype){
      // h[ka,kb] ka^+kb + h[kb,ka] kb^+ka
      const auto& qop = qops['B'][oper_pack(ka,kb)];
      qt2 += int1e.get(ka,kb)*qop + int1e.get(kb,ka)*qop.K(1); 
   }
   qops['H'][0] = qt2;
}

} // ctns

#endif
