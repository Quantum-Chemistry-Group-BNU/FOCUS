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

// kernel for <r'|ap^+aq|r>
void tns::oper_renorm_rightB_kernel(const qtensor3& bsite,
		                    const qtensor3& ksite,
		                    const qopers& cqops_ca,
		                    const qopers& cqops_c,
		                    const qopers& rqops_ca,
		                    const qopers& rqops_c,
		                    qopers& qops){
   qtensor3 qt3;
   qtensor2 qt2;
   // Ic * pR^+qR 
   for(const auto& rop_ca : rqops_ca){
      qt3 = contract_qt3_qt2_r(ksite,rop_ca);
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = rop_ca.msym;
      qt2.index[0] = rop_ca.index[0];
      qt2.index[1] = rop_ca.index[1];
      qops.push_back(qt2);
   }
   // pC^+ * qR and pR^+qC = -qC*pR^+
   for(const auto& cop_c : cqops_ca){
      for(const auto& rop_c : rqops_c){
	 // pC^+ * qR
	 auto qR = rop_c.transpose();
         qt3 = contract_qt3_qt2_r(ksite,qR);
         auto pCdag = cop_c.col_signed();
         qt3 = contract_qt3_qt2_c(qt3,pCdag);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
         qt2.msym = cop_c.msym - rop_c.msym;
         qt2.index[0] = cop_c.index[0];
         qt2.index[1] = rop_c.index[0];
         qops.push_back(qt2);
	 // pR^+qC = -qC*pR^+
         qt3 = contract_qt3_qt2_r(ksite,rop_c);
	 auto qCm = cop_c.transpose().col_signed(-1.0); 
         qt3 = contract_qt3_qt2_c(qt3,qCm);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
         qt2.msym = -cop_c.msym + rop_c.msym;
         qt2.index[0] = rop_c.index[0];
         qt2.index[1] = cop_c.index[0];
         qops.push_back(qt2);
      }
   }
   // pC^+qC * Ir
   for(const auto& cop_ca : cqops_ca){
      qt3 = contract_qt3_qt2_c(ksite,cop_ca); 
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = cop_ca.msym;
      qt2.index[0] = cop_ca.index[0];
      qt2.index[1] = cop_ca.index[1];
      qops.push_back(qt2);
   }
}
