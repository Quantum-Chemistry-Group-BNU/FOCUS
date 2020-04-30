#include "tns_qtensor.h"
#include "tns_oper.h"
#include "../core/matrix.h"

using namespace std;
using namespace linalg;
using namespace tns;

// kernel for <r'|ap^+|r>
void tns::oper_renorm_rightC_kernel(const qtensor3& bsite,
		                    const qtensor3& ksite,
		                    const qopers& cqops,
		                    const qopers& rqops,
		                    qopers& qops){
   qtensor3 qt3;
   qtensor2 qt2;
   // pR^+ 
   for(const auto& rop : rqops){
      qt3 = contract_qt3_qt2_r(ksite,rop);
      qt2 = contract_qt3_qt3_cr(bsite,qt3,true);
      qt2.msym = rop.msym;
      qt2.index[0] = rop.index[0];
      qops.push_back(qt2);
   }
   // pC^+ 
   for(const auto& cop : cqops){
      qt3 = contract_qt3_qt2_c(ksite,cop); 
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = cop.msym;
      qt2.index[0] = cop.index[0];
      qops.push_back(qt2);
   }
}
