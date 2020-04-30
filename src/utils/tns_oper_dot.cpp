#include "tns_oper.h"
#include "tns_qsym.h"
#include "../core/matrix.h"

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
   qtensor2 qt2;
   // c[0]
   // [[0. 0. 0. 0.]	|0_a,0_b>
   //  [0. 0. 0. 0.]	|0_a,1_b>
   //  [1. 0. 0. 0.]	|1_a,0_b>
   //  [0. 1. 0. 0.]]	|1_a,1_b>
   qt2.msym = qsym(1,1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[2],phys_sym[0])] = 1;
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[1])] = 1;
   qt2.index[0] = 2*k0;
   qops.push_back(qt2);
   // c[1]
   // [[ 0.  0.  0.  0.]
   //  [ 1.  0.  0.  0.]
   //  [ 0.  0.  0.  0.]
   //  [ 0.  0. -1.  0.]]
   qt2.msym = qsym(1,0);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[1],phys_sym[0])] = 1;
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[2])] = -1;
   qt2.index[0] = 2*k0+1;
   qops.push_back(qt2);
   return qops;
}

// 01 (c0<c1)
qopers tns::oper_dot_cc(const int k0){
   qopers qops;
   qtensor2 qt2;
   // c[0].dot(c[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [1. 0. 0. 0.]]
   qt2.msym = qsym(2,1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[0])] = 1;
   qt2.index[0] = 2*k0;
   qt2.index[1] = 2*k0+1;
   qops.push_back(qt2);
   return qops;
}

// 00,01,10,11
qopers tns::oper_dot_ca(const int k0){
   qopers qops;
   qtensor2 qt2;
   // c[0].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 1.]]
   qt2.msym = qsym(0,0);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[2],phys_sym[2])] = 1;
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qt2.index[0] = 2*k0;
   qt2.index[1] = 2*k0;
   qops.push_back(qt2);
   // c[0].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]]
   qt2.msym = qsym(0,1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[2],phys_sym[1])] = 1;
   qt2.index[0] = 2*k0;
   qt2.index[1] = 2*k0+1;
   qops.push_back(qt2);
   // c[1].dot(a[0])
   // [[0. 0. 0. 0.]
   //  [0. 0. 1. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qt2.msym = qsym(0,-1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[1],phys_sym[2])] = 1;
   qt2.index[0] = 2*k0+1;
   qt2.index[1] = 2*k0;
   qops.push_back(qt2);
   // c[1].dot(a[1])
   // [[0. 0. 0. 0.]
   //  [0. 1. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qt2.msym = qsym(0,0);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[1],phys_sym[1])] = 1;
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qt2.index[0] = 2*k0+1;
   qt2.index[1] = 2*k0+1;
   qops.push_back(qt2);
   return qops;
}

// 010,110
qopers tns::oper_dot_caa(const int k0){
   qopers qops;
   qtensor2 qt2;
   // c[0].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]]
   qt2.msym = qsym(-1,0);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[2],phys_sym[3])] = 1;
   qt2.index[0] = 2*k0;
   qt2.index[1] = 2*k0+1;
   qt2.index[2] = 2*k0;
   qops.push_back(qt2);
   // c[1].dot(a[1].dot(a[0]))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qt2.msym = qsym(-1,-1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[1],phys_sym[3])] = 1;
   qt2.index[0] = 2*k0+1;
   qt2.index[1] = 2*k0+1;
   qt2.index[2] = 2*k0;
   qops.push_back(qt2);
   return qops;
}
 
// 0110 (*<01||01>)
qopers tns::oper_dot_ccaa(const int k0){
   qopers qops;
   qtensor2 qt2;
   // c[0].dot(c[1].dot(a[1].dot(a[0])))
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 1.]]
   qt2.msym = qsym(0,0);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[3])] = 1;
   qt2.index[0] = 2*k0;
   qt2.index[1] = 2*k0+1;
   qt2.index[2] = 2*k0+1;
   qt2.index[3] = 2*k0;
   qops.push_back(qt2);
   return qops;
}
