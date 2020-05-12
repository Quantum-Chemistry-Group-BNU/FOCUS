#ifndef TNS_HAMILTONIAN_H
#define TNS_HAMILTONIAN_H

#include "../core/integral.h"
#include "tns_qtensor.h"
#include "tns_comb.h"
#include <vector>
#include <string>

namespace tns{

std::vector<double> get_Hdiag(const comb& icomb,
		 	      const comb_coord& p,
	              	      const integral::two_body& int2e,
	              	      const integral::one_body& int1e,
	    		      const double ecore,
			      const std::string scratch,
			      qtensor3& wf);

void get_Hx(double* y,
	    const double* x,
	    const comb& icomb,
 	    const comb_coord& p,
	    const integral::two_body& int2e,
	    const integral::one_body& int1e,
	    const double ecore,
	    const std::string scratch,
	    qtensor3& wf);

} // tns

#endif
