#ifndef TNS_OPER_H
#define TNS_OPER_H

#include "tns_comb.h"
#include "tns_qtensor.h"
#include "../core/integral.h"
#include <tuple>

namespace tns{

// build different types of operators specified by
// - coord (i,j)
// - kind (Symbol)

// 1. exact construction at type 0 site {|n>}
using qopers = std::map<int,qtensor2>;
qopers oper_dot_c();
qopers oper_dot_a();
qopers oper_dot_cc();
qopers oper_dot_ca();
qopers oper_dot_caa();
qopers oper_dot_ccaa();

// 7 kinds of operators:
//
// {C,A,B}
// Cp = ap^+
// Apq = ap^+aq^+
// Bpq = ap^+aq  
//
// {H,S,Q,P}
// H = hpq ap^+aq + 1/4<pq||sr> ap^+aq^+aras
// Sp = 1/2 hpq aq + <pq||sr> aq^+ ar as (r<s)
// Qps = <pq||sr> aq^+ ar
// Ppq = <pq||rs> ar as (r<s)

// 2. universal blocking code to deal with
//    - blocking at type 1,2 site (L/R) {|nr>}
//    - blocking at type 3 site (L/R) {|ur>}

} // tns

#endif
