#ifndef SWEEP_PRDM_H
#define SWEEP_PRDM_H

namespace ctns{

const double thresh_noise = 1.e-10;
extern const double thresh_noise;

// ZL@20210420:	
// Note that for perturbative noise, the signs are not properly treated as in 
// sweep_ham_onedot.h. We assume that they are not essential for recovering
// the losing quantum number! This proved to work for simple tests at least.

template <typename Tm>
void get_prdm_lc(const qtensor3<Tm>& wf, 
		 oper_dict<Tm>& lqops, 
		 oper_dict<Tm>& cqops, 
		 const double noise, 
		 qtensor2<Tm>& rdm){
   if(noise < thresh_noise) return;
   const bool dagger = true;
   auto t0 = tools::get_time();
   std::cout << "ctns::get_prdm_lc noise=" << std::setprecision(2) << noise;
   // local
   const auto& Hl = lqops['H'][0];
   const auto& Hc = cqops['H'][0];
   rdm += noise*oper_kernel_OIwf("lc",wf,Hl).merge_lc().get_rdm_row();
   rdm += noise*oper_kernel_IOwf("lc",wf,Hc,0).merge_lc().get_rdm_row();
   // a^L(+)|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row(); 
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lc().get_rdm_row();
   }
   // A^L|psi>
   for(const auto& op1A : lqops['A']){
      const auto& op1 = op1A.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lc().get_rdm_row();
   }
   // B^L|psi>
   for(const auto& op1B : lqops['B']){
      const auto& op1 = op1B.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lc().get_rdm_row();
   }
   // a^C(+)|psi>
   for(const auto& op2C : cqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,1).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,1,dagger).merge_lc().get_rdm_row();
   }
   // A^C|psi>
   for(const auto& op2A : cqops['A']){
      const auto& op2 = op2A.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0,dagger).merge_lc().get_rdm_row();
   }
   // B^C|psi>
   for(const auto& op2B : cqops['B']){
      const auto& op2 = op2B.second;
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0).merge_lc().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("lc",wf,op2,0,dagger).merge_lc().get_rdm_row();
   }
   // a+^La+^C|psi>, a+^La^C|psi>, a^La+|^Cpsi>, a^La^C|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : cqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1,op2,1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1,op2.H(),1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1.H(),op2,1).merge_lc().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lc",wf,op1.H(),op2.H(),1).merge_lc().get_rdm_row();	 
      }
   }
   auto t1 = tools::get_time();
   std::cout << " timing : " << tools::get_duration(t1-t0) << " s" << std::endl; 
}

template <typename Tm>
void get_prdm_lr(const qtensor3<Tm>& wf, 
		 oper_dict<Tm>& lqops, 
		 oper_dict<Tm>& rqops, 
		 const double noise, 
		 qtensor2<Tm>& rdm){
   if(noise < thresh_noise) return;
   const bool dagger = true;
   auto t0 = tools::get_time();
   std::cout << "ctns::get_prdm_lr noise=" << std::setprecision(2) << noise;
   // local
   const auto& Hl = lqops['H'][0];
   const auto& Hr = rqops['H'][0];
   rdm += noise*oper_kernel_OIwf("lr",wf,Hl).merge_lr().get_rdm_row();
   rdm += noise*oper_kernel_IOwf("lr",wf,Hr,0).merge_lr().get_rdm_row();
   // a^L+|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lr().get_rdm_row();
   }
   // A^L|psi>
   for(const auto& op1A : lqops['A']){
      const auto& op1 = op1A.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lr().get_rdm_row();
   }
   // B^L|psi>
   for(const auto& op1B : lqops['B']){
      const auto& op1 = op1B.second;
      rdm += noise*oper_kernel_OIwf("lc",wf,op1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_OIwf("lc",wf,op1,dagger).merge_lr().get_rdm_row();
   }
   // a^R+|psi>	
   for(const auto& op2C : rqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1,dagger).merge_lr().get_rdm_row();
   }
   // A^R|psi>
   for(const auto& op2A : rqops['A']){
      const auto& op2 = op2A.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0,dagger).merge_lr().get_rdm_row();
   }
   // B^R|psi>
   for(const auto& op2B : rqops['B']){
      const auto& op2 = op2B.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_lr().get_rdm_row();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0,dagger).merge_lr().get_rdm_row();
   }
   // a+^La+^R|psi>, a+^La^R|psi>, a^La+^R|psi>, a^La^R|psi>	
   for(const auto& op1C : lqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : rqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1,op2,1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1,op2.H(),1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1.H(),op2,1).merge_lr().get_rdm_row();	 
 	 rdm += noise*oper_kernel_OOwf("lr",wf,op1.H(),op2.H(),1).merge_lr().get_rdm_row();	 
      }
   }
   auto t1 = tools::get_time();
   std::cout << " timing : " << tools::get_duration(t1-t0) << " s" << std::endl; 
}

template <typename Tm>
void get_prdm_cr(const qtensor3<Tm>& wf, 
		 oper_dict<Tm>& cqops, 
		 oper_dict<Tm>& rqops, 
		 const double noise, 
		 qtensor2<Tm>& rdm){
   if(noise < thresh_noise) return;
   const bool dagger = true;
   auto t0 = tools::get_time();
   std::cout << "ctns::get_prdm_cr noise=" << std::setprecision(2) << noise;
   // local
   const auto& Hc = cqops['H'][0];
   const auto& Hr = rqops['H'][0];
   rdm += noise*oper_kernel_OIwf("cr",wf,Hc).merge_cr().get_rdm_col();
   rdm += noise*oper_kernel_IOwf("cr",wf,Hr,0).merge_cr().get_rdm_col();
   // a^C+|psi>
   for(const auto& op1C : cqops['C']){
      const auto& op1 = op1C.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_OIwf("cr",wf,op1,dagger).merge_cr().get_rdm_col();
   }
   // A^C|psi> 
   for(const auto& op1A : cqops['A']){
      const auto& op1 = op1A.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_OIwf("cr",wf,op1,dagger).merge_cr().get_rdm_col();
   }
   // B^C|psi>
   for(const auto& op1B : cqops['B']){
      const auto& op1 = op1B.second;
      rdm += noise*oper_kernel_OIwf("cr",wf,op1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_OIwf("cr",wf,op1,dagger).merge_cr().get_rdm_col();
   }
   // a^R+|psi>	
   for(const auto& op2C : rqops['C']){
      const auto& op2 = op2C.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,1,dagger).merge_cr().get_rdm_col();
   }
   // A^R|psi>
   for(const auto& op2A : rqops['A']){
      const auto& op2 = op2A.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0,dagger).merge_cr().get_rdm_col();
   }
   // B^R|psi>
   for(const auto& op2B : rqops['B']){
      const auto& op2 = op2B.second;
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0).merge_cr().get_rdm_col();
      rdm += noise*oper_kernel_IOwf("cr",wf,op2,0,dagger).merge_cr().get_rdm_col();
   }
   // a+^Ca+^R|psi>, a+^Ca^R|psi>, a^Ca+^R|psi>, a^Ca^R|psi>	
   for(const auto& op1C : cqops['C']){
      const auto& op1 = op1C.second;
      for(const auto& op2C : rqops['C']){
         const auto& op2 = op2C.second;
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1,op2,1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1,op2.H(),1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1.H(),op2,1).merge_cr().get_rdm_col();	 
 	 rdm += noise*oper_kernel_OOwf("cr",wf,op1.H(),op2.H(),1).merge_cr().get_rdm_col();	 
      }
   }
   auto t1 = tools::get_time();
   std::cout << " timing : " << tools::get_duration(t1-t0) << " s" << std::endl; 
}

} // ctns

#endif
