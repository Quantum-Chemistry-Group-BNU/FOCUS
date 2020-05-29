#include "tns_prdm.h"

void get_prdm_lc(const qtensor3& wf3, 
		 oper_dict& lqops, 
		 oper_dict& cqops, 
		 const double noise, 
		 qtensor2& rdm){
   // a^C+|psi>, A^C|psi>, B^C|psi>
   for(const auto& op2C : cqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("lc",wf,op2.T(),1).merge_lc().get_rdm_row();
   }
   for(const auto& op2A : cqops['A']){
      const auto& op2 = op2A.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("lc",wf,op2.T(),0).merge_lc().get_rdm_row();
   }
   for(const auto& op2B : cqops['B']){
      const auto& op2 = op2B.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0).merge_lc().get_rdm_row();
   }
   // a^L+|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1.T()).merge_lc().get_rdm_row();
   }
   // a^L+a^C|psi>, a^L+a^C|psi>, a^La^C+|psi>, a^La^C|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : cqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1,op2,1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1,op2.T(),1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1.T(),op2,1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1.T(),op2.T(),1).merge_lc().get_rdm_row();	 
      }
   }
   // PQ/AB^L|psi>
   bool ifPl = (lqops.find('P') != lqops.end());
   char key1 = ifPl? 'P' : 'A';
   char key2 = ifPl? 'Q' : 'B';
   for(const auto& op1P : lqops[key1]){
      const auto& op1 = op1P.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1.T()).merge_lc().get_rdm_row();
   }
   for(const auto& op1Q : lqops[key2]){
      const auto& op1 = op1Q.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row();
   }
}

void get_prdm_cr(const qtensor3& wf3, 
		 oper_dict& cqops, 
		 oper_dict& rqops, 
		 const double noise, 
		 qtensor2& rdm){
   // a^C+|psi>, A^C|psi>, B^C|psi>
   for(const auto& op1C : cqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_OIwf("cr",wf,op1.T()).merge_cr().get_rdm_col();
   }
   for(const auto& op1A : cqops['A']){
      const auto& op1 = op1A.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_OIwf("cr",wf,op1.T()).merge_cr().get_rdm_col();
   }
   for(const auto& op1B : cqops['B']){
      const auto& op1 = op1B.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
   }
   // a^R+|psi>	
   for(const auto& op2C : rqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2.T(),1).merge_cr().get_rdm_col();
   }
   // a^C+a^R|psi>, a^C+a^R|psi>, a^Ca^R+|psi>, a^Ca^R|psi>	
   for(const auto& op1C : cqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : rqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1,op2,1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1,op2.T(),1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1.T(),op2,1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1.T(),op2.T(),1).merge_cr().get_rdm_col();	 
      }
   }
   // PQ/AB^R|psi>
   bool ifPr = (rqops.find('P') != rqops.end());
   char key1 = ifPr? 'P' : 'A';
   char key2 = ifPr? 'Q' : 'B';
   for(const auto& op2P : rqops[key1]){
      const auto& op2 = op2P.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2.T(),0).merge_cr().get_rdm_col();
   }
   for(const auto& op2Q : rqops[key2]){
      const auto& op2 = op2Q.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_cr().get_rdm_col();
   }
}

void get_prdm_lr(const qtensor3& wf3, 
		 oper_dict& lqops, 
		 oper_dict& rqops, 
		 const double noise, 
		 qtensor2& rdm){
   // a^L+|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1.T()).merge_lr().get_rdm_row();
   }
   // a^R+|psi>	
   for(const auto& op2C : rqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2.T(),1).merge_lr().get_rdm_row();
   }
   // a^L+a^R|psi>, a^L+a^R|psi>, a^La^R+|psi>, a^La^R|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : rqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1,op2,1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1,op2.T(),1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1.T(),op2,1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1.T(),op2.T(),1).merge_lr().get_rdm_row();	 
      }
   }
   // PQ/AB^L|psi>
   bool ifPl = (lqops.find('P') != lqops.end());
   char key1 = ifPl? 'P' : 'A';
   char key2 = ifPl? 'Q' : 'B';
   for(const auto& op1P : lqops[key1]){
      const auto& op1 = op1P.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1.T()).merge_lr().get_rdm_row();
   }
   for(const auto& op1Q : lqops[key2]){
      const auto& op1 = op1Q.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
   }
   // PQ/AB^R|psi>
   bool ifPr = (rqops.find('P') != rqops.end());
   key1 = ifPr? 'P' : 'A';
   key2 = ifPr? 'Q' : 'B';
   for(const auto& op2P : rqops[key1]){
      const auto& op2 = op2P.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2.T(),0).merge_lr().get_rdm_row();
   }
   for(const auto& op2Q : rqops[key2]){
      const auto& op2 = op2Q.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_lr().get_rdm_row();
   }
}
