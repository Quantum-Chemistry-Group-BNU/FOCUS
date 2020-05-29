#ifndef TNS_DECIMATION_H
#define TNS_DECIMATION_H

#include "tns_comb.h"

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
		       const std::vector<qtensor3>& wfs,
		       double& dwt,
		       int& deff);

void decimation_twodot(comb& icomb, 
		       const comb_coord& p, 
		       const bool forward, 
		       const bool cturn, 
		       const int dcut,
		       const std::vector<qtensor4>& wfs,
		       double& dwt,
		       int& deff);
 
} // tns

#endif
