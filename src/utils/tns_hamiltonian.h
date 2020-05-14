#ifndef TNS_HAMILTONIAN_H
#define TNS_HAMILTONIAN_H

#include "../core/integral.h"
#include "tns_qtensor.h"
#include "tns_comb.h"
#include "tns_oper.h"
#include <vector>
#include <string>

namespace tns{

std::vector<double> get_Hdiag(oper_dict& cqops,
			      oper_dict& lqops,
			      oper_dict& rqops,
			      const double ecore,
			      qtensor3& wf);

void get_Hx(double* y,
	    const double* x,
	    const comb& icomb,
 	    const comb_coord& p,
	    oper_dict& cqops,
	    oper_dict& lqops,
	    oper_dict& rqops,
	    const integral::two_body& int2e,
	    const integral::one_body& int1e,
	    const double ecore,
	    qtensor3& wf);

} // tns

#endif
