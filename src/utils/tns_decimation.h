#ifndef TNS_DECIMATION_H
#define TNS_DECIMATION_H

#include "tns_comb.h"
#include "tns_oper.h"

namespace tns{

qtensor2 decimation_row(const qtensor2& rdm,
			const int dcut,
			double& dwt,
			int& deff,
			const bool trans=false);

void decimation_onedot(comb& icomb, 
		       const comb_coord& p, 
		       const bool forward, 
		       const bool cturn, 
		       const int dcut,
		       const linalg::matrix& vsol,
		       qtensor3& wf,
		       double& dwt,
		       int& deff,
		       const double noise, 
		       oper_dict& cqops,
		       oper_dict& lqops,
		       oper_dict& rqops);

void decimation_twodot(comb& icomb, 
		       const comb_coord& p, 
		       const bool forward, 
		       const bool cturn, 
		       const int dcut,
 		       const linalg::matrix& vsol,
		       qtensor4& wf,
		       double& dwt,
		       int& deff);

} // tns

#endif
