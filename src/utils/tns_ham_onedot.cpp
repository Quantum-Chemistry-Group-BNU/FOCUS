#include "tns_ham.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

vector<double> tns::get_onedot_Hdiag(oper_dict& cqops,
			             oper_dict& lqops,
			             oper_dict& rqops,
			             const double ecore,
			             qtensor3& wf){
   cout << "\ntns::get_onedot_Hdiag" << endl;
   // load Hl, Hc, Hr
   auto& Hc = cqops['H'][0];
   auto& Hl = lqops['H'][0];
   auto& Hr = rqops['H'][0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(auto& p : wf.qblocks){
      auto& blk = p.second;
      int mdim = blk.size();
      if(mdim > 0){
         auto& q = p.first;
         auto qm = get<0>(q);
         auto qr = get<1>(q);
         auto qc = get<2>(q);
	 auto& cblk = Hc.qblocks[make_pair(qm,qm)]; // central->mid
	 auto& lblk = Hl.qblocks[make_pair(qr,qr)]; // left->row
	 auto& rblk = Hr.qblocks[make_pair(qc,qc)]; // row->col
	 int rdim = blk[0].rows();
	 int cdim = blk[0].cols();
	 for(int m=0; m<mdim; m++){
	    for(int c=0; c<cdim; c++){
	       for(int r=0; r<rdim; r++){
	          blk[m](r,c) = ecore 
			      + lblk(r,r) 
			      + cblk(m,m)
			      + rblk(c,c);
	       } // r
	    } // c
	 } // m
      }
   } // qblocks
   vector<double> diag(wf.get_dim());
   wf.to_array(diag.data());
   return diag;
}

void tns::get_onedot_Hx(double* y,
		        const double* x,
		        const comb& icomb,
		        const comb_coord& p,
	                oper_dict& cqops,
		        oper_dict& lqops,
		        oper_dict& rqops,
		        const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const double ecore,
		        qtensor3& wf){
   bool debug = true;
   if(debug) cout << "\ntns::get_onedot_Hx" << endl;
   // const term
   wf.from_array(x);
   auto Hwf = ecore*wf;
   // construct H*wf
   // 1. local term: Hl,Hc,Hr
   Hwf += contract_qt3_qt2_l(wf,lqops['H'][0]);
/*
   Hwf += contract_qt3_qt2_c(wf,cqops['H'][0]);
   Hwf += contract_qt3_qt2_r(wf,rqops['H'][0]);
   // 2. local term: Il*HR
   Hwf += transform_ropH_wf(wf,p,cqops,rqops,int2e,int1e);
   // 3. pL^+ S_pL^R + h.c.
   for(const auto& lopC : lqops['C']){
      int pL = lopC.first;
      const auto& lop = lopC.second;
      // pL^+ (S_pL^R * wf)
      auto Swf = transform_ropS_wf(wf,p,cqops,rqops,int2e,int1e,0);
      Hwf += contract_qt3_qt2_l(wf,lop);
      // S_pL^R^+ * pL * wf
      auto Swf = transform_ropS_wf(wf,p,cqops,rqops,int2e,int1e,1);
      Hwf += contract_qt3_qt2_l(wf,lop.T());
   }
   // 4. qR^+ S_qR^L + h.c.
   for(const auto& ropC : rqops['C']){
      int qR = ropC.first;
      const auto& rop = ropC.second;
      auto Swf = transform_lopS_wf(wf,p,cqops,rqops,int2e,int1e,0);
      auto Swf = transform_lopS_wf(wf,p,cqops,rqops,int2e,int1e,1);
      // Hwf += 
   } 
   // ifAB:
   // 5. A: A_pLqL^L*P_pLqL^R + h.c.
   for(const auto& lopA : lqops['A']){
      const auto& lop = lopA.second;
      auto pq = oper_unpack(lopA.first);
      int pL = pq.first;
      int qL = pq.second; 
   }
   // 6. B: Q: B_pLsL^L*Q_pLsL^R
   for(const auto& lopB : lqops['B']){
      const auto& lop = lopB.second;
      auto ps = oper_unpack(lopB.first);
      int pL = ps.first;
      int sL = ps.second;
   } 
*/

   // finally copy back to y
   Hwf.to_array(y);
}
