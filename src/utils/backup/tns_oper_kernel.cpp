#include "tns_qtensor.h"
#include "tns_oper.h"

#include <iostream>

using namespace tns;

using namespace std;

// renorm Ic*Or
qtensor2 tns::oper_kernel_IcOr(const qtensor3& bsite,
	  	  	       const qtensor3& ksite,
			       const qtensor2& rop){
   auto qt3 = contract_qt3_qt2_r(ksite,rop);
   if(rop.index.size()%2 != 0){ 
      qt3 = qt3.mid_signed(); // due to fermionic exchange of |c> and Or
   }
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3,rop.sym);
   qt2.index = rop.index;
   return qt2;
}

// renorm Oc*Ir
qtensor2 tns::oper_kernel_OcIr(const qtensor3& bsite,
			       const qtensor3& ksite,
 			       const qtensor2& cop){
   auto qt3 = contract_qt3_qt2_c(ksite,cop); 
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3,cop.sym);
   qt2.index = cop.index;
   return qt2;
}

// renorm Oc*Or
qtensor2 tns::oper_kernel_OcOr(const qtensor3& bsite,
			       const qtensor3& ksite,
			       const qtensor2& cop,
			       const qtensor2& rop){
   auto qt3 = contract_qt3_qt2_r(ksite,rop);
   cout << "k:" << ksite.normF() << endl;
   cout << "b:" << bsite.normF() << endl;
   cout << "c:" << cop.normF() << endl;
   cout << "r:" << rop.normF() << endl;
   cout << "3:" << qt3.normF() << endl;

   if(rop.index.size()%2 == 0){
      qt3 = contract_qt3_qt2_c(qt3,cop);
   }else{
      auto cop_s = cop.col_signed();
      qt3 = contract_qt3_qt2_c(qt3,cop_s); 
   }
   
   cout << "3:" << qt3.normF() << endl;
   
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3,cop.sym+rop.sym);
   
   cout << "2:" << qt2.normF() << endl;

   exit(1);

   qt2.index = cop.index;
   copy(rop.index.begin(), rop.index.end(), back_inserter(qt2.index));
   return qt2;
}

// renorm Or*Oc = Oc*Or*(-1)^{p(Oc)*p(Or)}
qtensor2 tns::oper_kernel_OrOc(const qtensor3& bsite,
			       const qtensor3& ksite,
			       const qtensor2& rop,
			       const qtensor2& cop){
   auto qt3 = contract_qt3_qt2_r(ksite,rop);
   int pr = rop.index.size()%2;
   int pc = cop.index.size()%2;
   if(pr == 0){
      // er*ec = ec*er, er*oc = oc*er 
      qt3 = contract_qt3_qt2_c(qt3,cop);
   }else{
      if(pc == 0){
         // or*ec = ec*or
         auto cop_s = cop.col_signed();
	 qt3 = contract_qt3_qt2_c(qt3,cop_s);
      }else{
         // or*oc = -oc*or
         auto cop_s = cop.col_signed(-1);
	 qt3 = contract_qt3_qt2_c(qt3,cop_s);
      }
   }
   auto qt2 = contract_qt3_qt3_cr(bsite,qt3,cop.sym+rop.sym);
   qt2.index = rop.index;
   copy(cop.index.begin(), cop.index.end(), back_inserter(qt2.index));
   return qt2;
}
