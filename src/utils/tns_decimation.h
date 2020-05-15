#ifndef TNS_DECIMATION_H
#define TNS_DECIMATION_H

#include "tns_qtensor.h"
#include "tns_comb.h"
#include "tns_oper.h"

namespace tns{

qtensor2 decimation_row(const qtensor2& wf,
			const int Dcut);

qtensor2 decimation_col(const qtensor2& wf,
			const int Dcut);

} // tns

#endif
