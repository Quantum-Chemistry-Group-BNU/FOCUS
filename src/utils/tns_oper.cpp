#include "tns_comb.h" 
#include "tns_oper.h"

using namespace tns;
using namespace linalg;

// build different types of operators specified by
// - coord (i,j)
// - kind (ap,Apq,Bpq,Sp,Ppq,Qpq)

// 1. exact construction at type 0 site {|n>}

// 2. universal blocking code to deal with
//    - blocking at type 1,2 site (L/R) {|nr>}
//    - blocking at type 3 site (L/R) {|ur>}

matrix tns::get_Sij(const comb& bra, 
  		    const comb& ket){

}

matrix tns::get_Hij(const comb& bra, 
		    const comb& ket,
		    const integral::two_body& int2e,
		    const integral::one_body& int1e,
		    const double ecore){

}
