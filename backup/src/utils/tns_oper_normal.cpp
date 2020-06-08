#include "../settings/global.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// kernel for computing renormalized ap^+
void tns::oper_renorm_opC(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opC" << endl;
   auto t0 = global::get_time();
   // 1. pC^+*Ir 
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      auto Hwf = oper_kernel_OIwf(superblock,ksite,op1);
      qops['C'][p1] = oper_kernel_renorm(superblock,bsite,Hwf);
   }
   // 2. Ic*pR^+ 
   for(const auto& op2C : qops2['C']){
      int p2 = op2C.first;
      const auto& op2 = op2C.second;
      auto Hwf = oper_kernel_IOwf(superblock,ksite,op2,1);
      qops['C'][p2] = oper_kernel_renorm(superblock,bsite,Hwf); 
   }
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::oper_renorm_opC : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}

// kernel for computing renormalized Apq=ap^+aq^+
void tns::oper_renorm_opA(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opA" << endl;
   auto t0 = global::get_time();
   // 1. pC^+qC^+ * Ir
   for(const auto& op1A : qops1['A']){
      int pq1 = op1A.first;
      const auto& op1 = op1A.second;
      auto Hwf = oper_kernel_OIwf(superblock,ksite,op1);
      qops['A'][pq1] = oper_kernel_renorm(superblock,bsite,Hwf);
   }
   // 2. Ic * pR^+qR^+ (p<q) 
   for(const auto& op2A : qops2['A']){
      int pq2 = op2A.first;
      const auto& op2 = op2A.second;
      auto Hwf = oper_kernel_IOwf(superblock,ksite,op2,0);
      qops['A'][pq2] = oper_kernel_renorm(superblock,bsite,Hwf);
   }
   // 3. pC^+ * qR^+
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      for(const auto& op2C : qops2['C']){
	 int p2 = op2C.first;
	 const auto& op2 = op2C.second;
         auto Hwf = oper_kernel_OOwf(superblock,ksite,op1,op2,1);
	 auto qt2 = oper_kernel_renorm(superblock,bsite,Hwf);
	 // only store Apq where p<q;
	 if(p1 < p2){
	    qops['A'][oper_pack(p1,p2)] = qt2; 
	 }else{
	    qops['A'][oper_pack(p2,p1)] = -qt2;
	 }
      }
   }
   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::oper_renorm_opA : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}

// kernel for computing renormalized ap^+aq
void tns::oper_renorm_opB(const string& superblock,
			  const qtensor3& bsite,
			  const qtensor3& ksite,
			  oper_dict& qops1,
			  oper_dict& qops2,
			  oper_dict& qops,
			  const bool debug){
   if(debug) cout << "tns::oper_renorm_opB" << endl;
   auto t0 = global::get_time();
   // 1. pC^+qC * Ir
   for(const auto& op1B : qops1['B']){
      int pq1 = op1B.first;
      const auto& op1 = op1B.second;
      auto Hwf = oper_kernel_OIwf(superblock,ksite,op1);
      qops['B'][pq1] = oper_kernel_renorm(superblock,bsite,Hwf);
   }
   // 2. Ic * pR^+qR 
   for(const auto& op2B : qops2['B']){
      int pq2 = op2B.first;
      const auto& op2 = op2B.second;
      auto Hwf = oper_kernel_IOwf(superblock,ksite,op2,0);
      qops['B'][pq2] = oper_kernel_renorm(superblock,bsite,Hwf);
   }
   // 3. pC^+ * qR and pR^+*qC = -qC*pR^+
   for(const auto& op1C : qops1['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      for(const auto& op2C : qops2['C']){
	 int p2 = op2C.first;
	 const auto& op2 = op2C.second;
	 auto Hwf = oper_kernel_OOwf(superblock,ksite,op1,op2.T(),1);
	 qops['B'][oper_pack(p1,p2)] =  oper_kernel_renorm(superblock,bsite,Hwf);
	 Hwf = oper_kernel_OOwf(superblock,ksite,op1.T(),op2,1);
	 qops['B'][oper_pack(p2,p1)] =  -oper_kernel_renorm(superblock,bsite,Hwf);
      }
   }
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::oper_renorm_opB : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}
