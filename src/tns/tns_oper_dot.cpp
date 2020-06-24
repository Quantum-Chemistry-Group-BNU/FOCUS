#include "tns_qsym.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

// Note:
// we use the convention that p1+p2+p3+q3q2q1 where p1<p2<p3 and q3>q2>q1
// that is index in the middle is larger than that close to the boundary.
// This is different from the ordering used in onstate.h

// <l'|ap^+|l>: 0,1
void tns::oper_dot_C(const int k0, oper_dict& qops){
   int ka = 2*k0, kb = ka+1;
   // c[0]
   // [[0. 0. 0. 0.]	|0_a,0_b>
   //  [0. 0. 0. 0.]	|0_a,1_b>
   //  [1. 0. 0. 0.]	|1_a,0_b>
   //  [0. 1. 0. 0.]]	|1_a,1_b>
   qtensor2 qt2a(qsym(1,1),phys_qsym_space,phys_qsym_space);
   qt2a.qblocks[make_pair(phys_sym[2],phys_sym[0])] = 1;
   qt2a.qblocks[make_pair(phys_sym[3],phys_sym[1])] = 1;
   qops['C'][ka] = qt2a;
   // c[1]
   // [[ 0.  0.  0.  0.]
   //  [ 1.  0.  0.  0.]
   //  [ 0.  0.  0.  0.]
   //  [ 0.  0. -1.  0.]]
   qtensor2 qt2b(qsym(1,0),phys_qsym_space,phys_qsym_space);
   qt2b.qblocks[make_pair(phys_sym[1],phys_sym[0])] = 1;
   qt2b.qblocks[make_pair(phys_sym[3],phys_sym[2])] = -1;
   qops['C'][kb] = qt2b;
}

// 01 (c0<c1)
void tns::oper_dot_A(const int k0, oper_dict& qops){
   int ka = 2*k0, kb = ka+1;
   // c[0].dot(c[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [1. 0. 0. 0.]]
   qtensor2 qt2ab(qsym(2,1),phys_qsym_space,phys_qsym_space);
   qt2ab.qblocks[make_pair(phys_sym[3],phys_sym[0])] = 1;
   qops['A'][oper_pack(ka,kb)] = qt2ab;
}

// 00,01,(10),11
void tns::oper_dot_B(const int k0, oper_dict& qops){
   int ka = 2*k0, kb = ka+1;
   // c[0].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2aa(qsym(0,0),phys_qsym_space,phys_qsym_space);
   qt2aa.qblocks[make_pair(phys_sym[2],phys_sym[2])] = 1;
   qt2aa.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops['B'][oper_pack(ka,ka)] = qt2aa;
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2ab(qsym(0,1),phys_qsym_space,phys_qsym_space);
   qt2ab.qblocks[make_pair(phys_sym[2],phys_sym[1])] = 1;
   qops['B'][oper_pack(ka,kb)] = qt2ab;
   // c[1].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2ba(qsym(0,-1),phys_qsym_space,phys_qsym_space);
   qt2ba.qblocks[make_pair(phys_sym[1],phys_sym[2])] = 1;
   qops['B'][oper_pack(kb,ka)] = qt2ba;
   // c[1].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2bb(qsym(0,0),phys_qsym_space,phys_qsym_space);
   qt2bb.qblocks[make_pair(phys_sym[1],phys_sym[1])] = 1;
   qt2bb.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops['B'][oper_pack(kb,kb)] = qt2bb;
}
 
// build local S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
void tns::oper_dot_S(const int k0,
		     const integral::two_body<double>& int2e,
		     const integral::one_body<double>& int1e,
		     const vector<int>& psupp,
		     oper_dict& qops){
   int ka = 2*k0, kb = ka+1;
   // caa = 010,110
   // c[0].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2aba(qsym(-1,0),phys_qsym_space,phys_qsym_space);
   qt2aba.qblocks[make_pair(phys_sym[2],phys_sym[3])] = 1;
   // c[1].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2bba(qsym(-1,-1),phys_qsym_space,phys_qsym_space);
   qt2bba.qblocks[make_pair(phys_sym[1],phys_sym[3])] = 1;
   // S_{p}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
   for(int korb_p : psupp){
      int pa = 2*korb_p, pb = pa+1;
      // SpLa
      qtensor2 Spa = 0.5*int1e.get(pa,ka)*qops['C'][ka].T()	    
      	           + int2e.get(pa,kb,ka,kb)*qt2bba; // qrs=110
      qops['S'][pa] = Spa;
      // SpLb
      qtensor2 Spb = 0.5*int1e.get(pb,kb)*qops['C'][kb].T()
      	           + int2e.get(pb,ka,ka,kb)*qt2aba; // qrs=010
      qops['S'][pb] = Spb;
   } // p
}

// build local H^C = hpq ap^+aq + <pq||sr> ap^+aq^+aras [p<q,r>s]
void tns::oper_dot_H(const int k0,
		     const integral::two_body<double>& int2e,
		     const integral::one_body<double>& int1e,
		     oper_dict& qops){
   int ka = 2*k0, kb = ka+1;
   // 0110 (*<01||01>)
   // c[0].dot(c[1].dot(a[1].dot(a[0])))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2abba(qsym(0,0),phys_qsym_space,phys_qsym_space);
   qt2abba.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops['H'][0] = int1e.get(ka,ka)*qops['B'][oper_pack(ka,ka)]
		+ int1e.get(kb,kb)*qops['B'][oper_pack(kb,kb)]
		+ int2e.get(ka,kb,ka,kb)*qt2abba;
}
