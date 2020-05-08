#ifndef TNS_ALG_H
#define TNS_ALG_H

#include "tns_comb.h"
#include "../core/integral.h"

namespace tns{

linalg::matrix get_Smat(const comb& bra, 
  		        const comb& ket);

linalg::matrix get_Hmat(const comb& bra, 
		        const comb& ket,
		        const integral::two_body& int2e,
		        const integral::one_body& int1e,
		        const double ecore,
		        const std::string scratch);

} // tns

#endif
