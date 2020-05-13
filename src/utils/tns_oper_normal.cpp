#include "../settings/global.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"

using namespace std;
using namespace linalg;
using namespace tns;

void tns::oper_renorm_ropC(const qtensor3& bsite,
			   const qtensor3& ksite,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropC" << endl;
   auto t0 = global::get_time();
   
   // kernel for computing renormalized ap^+
   // 1. Ic*pR^+ 
   for(const auto& ropC : rqops['C']){
      qops['C'][ropC.first] = oper_kernel_IcOr(bsite,ksite,ropC.second,1); 
   }
   // 2. pC^+*Ir 
   for(const auto& copC : cqops['C']){
      qops['C'][copC.first] = oper_kernel_OcIr(bsite,ksite,copC.second);
   }
   
   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropC : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_ropA(const qtensor3& bsite,
			   const qtensor3& ksite,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropA" << endl;
   auto t0 = global::get_time();
   
   // kernel for computing renormalized Apq=ap^+aq^+
   // 1. Ic * pR^+qR^+ (p<q) 
   for(const auto& ropA : rqops['A']){
      qops['A'][ropA.first] = oper_kernel_IcOr(bsite,ksite,ropA.second,0);
   }
   // 2. pC^+qC^+ * Ir
   for(const auto& copA : cqops['A']){
      qops['A'][copA.first] = oper_kernel_OcIr(bsite,ksite,copA.second); 
   }
   // 3. pC^+ * qR^+
   for(const auto& copC : cqops['C']){
      int pC = copC.first;
      const auto& cop = copC.second;
      for(const auto& ropC : rqops['C']){
	 int qR = ropC.first;
	 const auto& rop = ropC.second;
	 // only store Apq where node[p]<node[q]
	 qops['A'][oper_pack(pC,qR)] = oper_kernel_OcOr(bsite,ksite,cop,rop,1);
      }
   }

   auto t1 = global::get_time();
   if(debug){
      cout << "timing for tns::renorm_ropA : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}

void tns::oper_renorm_ropB(const qtensor3& bsite,
			   const qtensor3& ksite,
			   oper_dict& cqops,
			   oper_dict& rqops,
			   oper_dict& qops,
			   const bool debug){
   if(debug) cout << "tns::oper_renorm_ropB" << endl;
   auto t0 = global::get_time();
   
   // kernel for computing renormalized ap^+aq
   // Ic * pR^+qR 
   for(const auto& ropB : rqops['B']){
      qops['B'][ropB.first] = oper_kernel_IcOr(bsite,ksite,ropB.second,0);
   }
   // pC^+qC * Ir
   for(const auto& copB : cqops['B']){
      qops['B'][copB.first] = oper_kernel_OcIr(bsite,ksite,copB.second);
   }
   // pC^+ * qR and pR^+*qC = -qC*pR^+
   for(const auto& copC : cqops['C']){
      int pC = copC.first;
      const auto& cop = copC.second;
      for(const auto& ropC : rqops['C']){
	 int pR = ropC.first;
	 const auto& rop = ropC.second;
	 qops['B'][oper_pack(pC,pR)] =  oper_kernel_OcOr(bsite,ksite,cop,rop.T(),1);
	 qops['B'][oper_pack(pR,pC)] = -oper_kernel_OcOr(bsite,ksite,cop.T(),rop,1);
      }
   }

   auto t1 = global::get_time();
   if(debug){ 
      cout << "timing for tns::renorm_ropB : " << setprecision(2) 
           << global::get_duration(t1-t0) << " s" << endl;
   }
}
