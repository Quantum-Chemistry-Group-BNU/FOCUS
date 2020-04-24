#include "tns_sweep.h"

using namespace tns;

// build different types of operators specified by
// - coord (i,j)
// - kind (ap,Apq,Bpq,Sp,Ppq,Qpq)

// 1. exact construction at type 0 site {|n>}

// 2. universal blocking code to deal with
//    - blocking at type 1,2 site (L/R) {|nr>}
//    - blocking at type 3 site (L/R) {|ur>}

