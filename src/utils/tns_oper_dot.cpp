#include "../core/matrix.h"
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
qopers tns::oper_dot_c(const int k0){
   qopers qops;
   // c[0]
   // [[0. 0. 0. 0.]	|0_a,0_b>
   //  [0. 0. 0. 0.]	|0_a,1_b>
   //  [1. 0. 0. 0.]	|1_a,0_b>
   //  [0. 1. 0. 0.]]	|1_a,1_b>
   qtensor2 qt2a(qsym(1,1),phys_qsym_space,phys_qsym_space,1);
   qt2a.index[0] = 2*k0;
   qt2a.qblocks[make_pair(phys_sym[2],phys_sym[0])] = 1;
   qt2a.qblocks[make_pair(phys_sym[3],phys_sym[1])] = 1;
   qops.push_back(qt2a);
   // c[1]
   // [[ 0.  0.  0.  0.]
   //  [ 1.  0.  0.  0.]
   //  [ 0.  0.  0.  0.]
   //  [ 0.  0. -1.  0.]]
   qtensor2 qt2b(qsym(1,0),phys_qsym_space,phys_qsym_space,1);
   qt2b.index[0] = 2*k0+1;
   qt2b.qblocks[make_pair(phys_sym[1],phys_sym[0])] = 1;
   qt2b.qblocks[make_pair(phys_sym[3],phys_sym[2])] = -1;
   qops.push_back(qt2b);
   return qops;
}

// 01 (c0<c1)
qopers tns::oper_dot_cc(const int k0){
   qopers qops;
   // c[0].dot(c[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [1. 0. 0. 0.]]
   qtensor2 qt2ab(qsym(2,1),phys_qsym_space,phys_qsym_space,2);
   qt2ab.index[0] = 2*k0;
   qt2ab.index[1] = 2*k0+1;
   qt2ab.qblocks[make_pair(phys_sym[3],phys_sym[0])] = 1;
   qops.push_back(qt2ab);
   return qops;
}

// 00,01,(10),11
qopers tns::oper_dot_ca(const int k0){
   qopers qops;
   // c[0].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2aa(qsym(0,0),phys_qsym_space,phys_qsym_space,2);
   qt2aa.index[0] = 2*k0;
   qt2aa.index[1] = 2*k0;
   qt2aa.qblocks[make_pair(phys_sym[2],phys_sym[2])] = 1;
   qt2aa.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops.push_back(qt2aa);
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2ab(qsym(0,1),phys_qsym_space,phys_qsym_space,2);
   qt2ab.index[0] = 2*k0;
   qt2ab.index[1] = 2*k0+1;
   qt2ab.qblocks[make_pair(phys_sym[2],phys_sym[1])] = 1;
   qops.push_back(qt2ab);
   // c[1].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2ba(qsym(0,-1),phys_qsym_space,phys_qsym_space,2);
   qt2ba.index[0] = 2*k0+1;
   qt2ba.index[1] = 2*k0;
   qt2ba.qblocks[make_pair(phys_sym[1],phys_sym[2])] = 1;
   qops.push_back(qt2ba);
   // c[1].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2bb(qsym(0,0),phys_qsym_space,phys_qsym_space,2);
   qt2bb.index[0] = 2*k0+1;
   qt2bb.index[1] = 2*k0+1;
   qt2bb.qblocks[make_pair(phys_sym[1],phys_sym[1])] = 1;
   qt2bb.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops.push_back(qt2bb);
   return qops;
}

// 010,110
qopers tns::oper_dot_caa(const int k0){
   qopers qops;
   // c[0].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2aba(qsym(-1,0),phys_qsym_space,phys_qsym_space,3);
   qt2aba.index[0] = 2*k0;
   qt2aba.index[1] = 2*k0+1;
   qt2aba.index[2] = 2*k0;
   qt2aba.qblocks[make_pair(phys_sym[2],phys_sym[3])] = 1;
   qops.push_back(qt2aba);
   // c[1].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qtensor2 qt2bba(qsym(-1,-1),phys_qsym_space,phys_qsym_space,3);
   qt2bba.index[0] = 2*k0+1;
   qt2bba.index[1] = 2*k0+1;
   qt2bba.index[2] = 2*k0;
   qt2bba.qblocks[make_pair(phys_sym[1],phys_sym[3])] = 1;
   qops.push_back(qt2bba);
   return qops;
}
 
// 0110 (*<01||01>)
qopers tns::oper_dot_ccaa(const int k0){
   qopers qops;
   // c[0].dot(c[1].dot(a[1].dot(a[0])))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qtensor2 qt2abba(qsym(0,0),phys_qsym_space,phys_qsym_space,4);
   qt2abba.index[0] = 2*k0;
   qt2abba.index[1] = 2*k0+1;
   qt2abba.index[2] = 2*k0+1;
   qt2abba.index[3] = 2*k0;
   qt2abba.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qops.push_back(qt2abba);
   return qops;
}

// build local S_{p_L}^C = 1/2 hpq aq + <pq||sr> aq^+aras [r>s]
void tns::oper_dot_rightS_loc(const int k,
			      const vector<int>& lsupp,
			      const qopers& cqops_c,
			      const qopers& cqops_caa,
			      const integral::two_body& int2e,
		              const integral::one_body& int1e,
			      qopers& cqops_S){
   int ka = 2*k, kb = ka+1;
   for(int korb_p : lsupp){
      int pa = 2*korb_p, pb = pa+1;
      // SpLa
      qtensor2 Spa = 0.5*int1e.get(pa,ka)*cqops_c[0].transpose()
      	           + int2e.getAnti(pa,kb,ka,kb)*cqops_caa[1]; // qrs=110
      Spa.index.resize(1);
      Spa.index[0] = pa;
      cqops_S.push_back(Spa);
      // SpLb
      qtensor2 Spb = 0.5*int1e.get(pb,kb)*cqops_c[1].transpose()
      	           + int2e.getAnti(pa,ka,ka,kb)*cqops_caa[0]; // qrs=010
      Spb.index.resize(1);
      Spb.index[0] = pb;
      cqops_S.push_back(Spb);
   } // p
}
