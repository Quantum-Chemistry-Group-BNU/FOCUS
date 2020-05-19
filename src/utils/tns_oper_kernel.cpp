#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// O1*I2|psi>
qtensor3 tns::oper_kernel_OIwf(const string& superblock,
			       const qtensor3& ksite,
 			       const qtensor2& o1){
   qtensor3 qt3;
   if(superblock == "lc"){
      qt3 = contract_qt3_qt2_l(ksite,o1);
   }else if(superblock == "lr"){
      qt3 = contract_qt3_qt2_l(ksite,o1);
   }else if(superblock == "cr"){
      qt3 = contract_qt3_qt2_c(ksite,o1);
   }else{
      cout << "error: no such case in oper_kernel_OIwf! superblock=" 
	   << superblock << endl;
      exit(1);
   }
   return qt3;
}

// I1*O2|psi>
qtensor3 tns::oper_kernel_IOwf(const string& superblock,
			       const qtensor3& ksite,
 			       const qtensor2& o2,
			       const bool po2){
   qtensor3 qt3;
   if(superblock == "lc"){
      qt3 = contract_qt3_qt2_c(ksite,o2);
      if(po2) qt3 = qt3.row_signed();
   }else if(superblock == "lr"){
      qt3 = contract_qt3_qt2_r(ksite,o2);
      if(po2) qt3 = qt3.row_signed();
   }else if(superblock == "cr"){
      qt3 = contract_qt3_qt2_r(ksite,o2);
      if(po2) qt3 = qt3.mid_signed();
   }else{
      cout << "error: no such case in oper_kernel_IOwf! superblock=" 
	   << superblock << endl;
      exit(1);
   }
   return qt3;
}

// O1*O2|psi>
qtensor3 tns::oper_kernel_OOwf(const string& superblock,
			       const qtensor3& ksite,
 			       const qtensor2& o1,
 			       const qtensor2& o2,
			       const bool po2){
   auto qt3 = oper_kernel_IOwf(superblock, ksite, o2, po2);
   return oper_kernel_OIwf(superblock, qt3, o1);
}

// <bra|O|ket>
qtensor2 tns::oper_kernel_renorm(const string& superblock,
			         const qtensor3& bsite,
				 const qtensor3& ksite){
   qtensor2 qt2;
   if(superblock == "lc"){
      qt2 = contract_qt3_qt3_lc(bsite,ksite);
   }else if(superblock == "lr"){
      qt2 = contract_qt3_qt3_lr(bsite,ksite);
   }else if(superblock == "cr"){
      qt2 = contract_qt3_qt3_cr(bsite,ksite);
   }else{
      cout << "error: no such case in oper_kernel_OIwf! superblock=" 
	   << superblock << endl;
      exit(1);
   }
   return qt2;
}
