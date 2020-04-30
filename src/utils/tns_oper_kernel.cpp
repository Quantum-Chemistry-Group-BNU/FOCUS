#include "tns_qtensor.h"
#include "tns_oper.h"
#include "../core/matrix.h"

using namespace std;
using namespace linalg;
using namespace tns;

// kernel for <r'|ap^+|r>
void tns::oper_kernel_rightC(const qtensor3& bsite,
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
void tns::oper_kernel_rightB(const qtensor3& bsite,
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
   // pC^+ * qR and pR^+qC
   for(const auto& cop_c : cqops_c){
      for(const auto& rop_c : rqops_c){
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

// kernel for <r'|ap^+aq^+|r> (p<q)
void tns::oper_kernel_rightA(const qtensor3& bsite,
		             const qtensor3& ksite,
		             const qopers& cqops_cc,
		             const qopers& cqops_c,
		             const qopers& rqops_cc,
		             const qopers& rqops_c,
		             qopers& qops){
   qtensor3 qt3;
   qtensor2 qt2;
   // Ic * pR^+qR^+ (p<q) 
   for(const auto& rop_cc : rqops_cc){
      qt3 = contract_qt3_qt2_r(ksite,rop_cc);
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = rop_cc.msym;
      qt2.index[0] = rop_cc.index[0];
      qt2.index[1] = rop_cc.index[1];
      qops.push_back(qt2);
   }
   // pC^+ * qR^+ 
   for(const auto& cop_c : cqops_c){
      for(const auto& rop_c : rqops_c){
	 // pC^+ * qR^+
         qt3 = contract_qt3_qt2_r(ksite,rop_c);
         auto pCdag = cop_c.col_signed();
         qt3 = contract_qt3_qt2_c(qt3,pCdag);
         qt2 = contract_qt3_qt3_cr(bsite,qt3);
         qt2.msym = cop_c.msym + rop_c.msym;
         qt2.index[0] = cop_c.index[0];
         qt2.index[1] = rop_c.index[0];
         qops.push_back(qt2);
      }
   }
   // pC^+qC^+ * Ir
   for(const auto& cop_cc : cqops_cc){
      qt3 = contract_qt3_qt2_c(ksite,cop_cc); 
      qt2 = contract_qt3_qt3_cr(bsite,qt3);
      qt2.msym = cop_cc.msym;
      qt2.index[0] = cop_cc.index[0];
      qt2.index[1] = cop_cc.index[1];
      qops.push_back(qt2);
   }
}
