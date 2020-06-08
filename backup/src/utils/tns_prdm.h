#ifndef TNS_PRDM_H
#define TNS_PRDM_H

#include "tns_qtensor.h"
#include "tns_oper.h"

namespace tns{

void get_prdm_lc(const qtensor3& wf, 
		 oper_dict& lqops, 
		 oper_dict& cqops, 
		 const double noise,
		 qtensor2& rdm);

void get_prdm_cr(const qtensor3& wf, 
		 oper_dict& cqops, 
		 oper_dict& rqops, 
		 const double noise,
		 qtensor2& rdm);

void get_prdm_lr(const qtensor3& wf, 
		 oper_dict& lqops, 
		 oper_dict& rqops, 
		 const double noise,
		 qtensor2& rdm);

} // tns

#endif
