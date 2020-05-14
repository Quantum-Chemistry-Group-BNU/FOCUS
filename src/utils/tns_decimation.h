#ifndef TNS_DECIMATION_H
#define TNS_DECIMATION_H

#include "../core/matrix.h"
#include "tns_qtensor.h"
#include "tns_comb.h"
#include "tns_oper.h"

namespace tns{

qtensor3 decimation_onedot(const comb& icomb,
		 	   const comb_coord& p,
			   qtensor3& wf,
			   linalg::matrix& vsol);

} // tns

#endif
