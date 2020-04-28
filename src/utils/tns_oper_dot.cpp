#include "tns_oper.h"
#include "tns_qsym.h"
#include "../core/matrix.h"

using namespace std;
using namespace linalg;
using namespace tns;

// <l'|ap^+|l>: 0,1
qopers tns::oper_dot_c(){
   qopers qops;
   qtensor2 qt2;
   // c[0]
   // [[0. 0. 0. 0.]
   //  [0. 0. 0. 0.]
   //  [1. 0. 0. 0.]
   //  [0. 1. 0. 0.]]
   qt2.msym = qsym(1,1);
   qt2.qrow = phys_qsym_space;
   qt2.qcol = phys_qsym_space;
   qt2.init_qblocks();
   qt2.qblocks[make_pair(phys_sym[2],phys_sym[0])] = 1;
   qt2.qblocks[make_pair(phys_sym[3],phys_sym[1])] = 1;
   qops[0] = qt2;
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
   qops[1] = qt2;
   return qops;
}

qopers tns::oper_dot_a(){
   qopers qops = oper_dot_c();
   // a[0]
   // [[0. 0. 1. 0.]
   //  [0. 0. 0. 1.]
   //  [0. 0. 0. 0.]
   //  [0. 0. 0. 0.]]
   qops[0] = qops[0].transpose();
   // a[1]
   // [[ 0.  1.  0.  0.]
   //  [ 0.  0.  0.  0.]
   //  [ 0.  0.  0. -1.]
   //  [ 0.  0.  0.  0.]]
   qops[1] = qops[1].transpose();
   return qops;
}

// 01 (c0<c1)
qopers tns::oper_dot_cc(){
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
   qops[0] = qt2;
   return qops;
}

// 00,01,10,11
qopers tns::oper_dot_ca(){
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
   qops[0] = qt2;
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
   qops[1] = qt2;
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
   qops[2] = qt2;
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
   qops[3] = qt2;
   return qops;
}

// 010,110
qopers tns::oper_dot_caa(){
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
   qops[0] = qt2;
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
   qops[1] = qt2;
   return qops;
}
 
// 0110 (*<01||01>)
qopers tns::oper_dot_ccaa(){
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
   qops[0] = qt2;
   return qops;
}
