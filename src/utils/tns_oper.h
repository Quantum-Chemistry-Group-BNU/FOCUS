#ifndef TNS_OPER_H
#define TNS_OPER_H

#include "tns_comb.h"
#include "../core/integral.h"

namespace tns{

linalg::matrix get_Sij(const comb& bra, 
  		       const comb& ket);

linalg::matrix get_Hij(const comb& bra, 
		       const comb& ket,
		       const integral::two_body& int2e,
		       const integral::one_body& int1e,
		       const double ecore);

} // tns

#endif
