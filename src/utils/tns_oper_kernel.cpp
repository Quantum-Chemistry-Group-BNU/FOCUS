#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace tns;

// renorm Oc*Ir
qtensor2 tns::oper_kernel_OcIr(const qtensor3& bsite,
			       const qtensor3& ksite,
 			       const qtensor2& cop){
   auto qt3 = contract_qt3_qt2_c(ksite,cop); 
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3);
   return qt2;
}

// renorm Ic*Or
qtensor2 tns::oper_kernel_IcOr(const qtensor3& bsite,
	  	  	       const qtensor3& ksite,
			       const qtensor2& rop,
			       const int prop){
   auto qt3 = contract_qt3_qt2_r(ksite,rop);
   // due to fermionic exchange of |c> and Or
   if(prop != 0) qt3 = qt3.mid_signed(); 
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3);
   return qt2;
}


// renorm Oc*Or
qtensor2 tns::oper_kernel_OcOr(const qtensor3& bsite,
			       const qtensor3& ksite,
			       const qtensor2& cop,
			       const qtensor2& rop,
			       const int prop){
   auto qt3 = contract_qt3_qt2_r(ksite,rop);
   if(prop == 0){
      qt3 = contract_qt3_qt2_c(qt3,cop);
   }else{	
      auto cop_s = cop.col_signed();
      qt3 = contract_qt3_qt2_c(qt3,cop_s); 
   }
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3);
   return qt2;
}
