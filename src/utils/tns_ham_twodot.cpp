#include "tns_ham.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

vector<double> tns::get_twodot_Hdiag(oper_dict& c1qops,
			             oper_dict& c2qops,
			             oper_dict& lqops,
			             oper_dict& rqops,
		        	     const integral::two_body& int2e,
			             const double ecore,
			             qtensor4& wf){
   cout << "\ntns::get_twodot_Hdiag" << endl;
   // load Hl, Hc, Hr
   auto& Hc1 = c1qops['H'][0];
   auto& Hc2 = c2qops['H'][0];
   auto& Hl = lqops['H'][0];
   auto& Hr = rqops['H'][0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(auto& p : wf.qblocks){
      auto& blk = p.second;
      int mvdim = blk.size();
      if(mvdim > 0){
         auto& q = p.first;
         auto qm = get<0>(q);
         auto qv = get<1>(q);
         auto qr = get<2>(q);
         auto qc = get<3>(q);
	 auto& c1blk = Hc1.qblocks[make_pair(qm,qm)]; // central0->mid
	 auto& c2blk = Hc2.qblocks[make_pair(qv,qv)]; // central1->ver
	 auto& lblk = Hl.qblocks[make_pair(qr,qr)]; // left->row
	 auto& rblk = Hr.qblocks[make_pair(qc,qc)]; // row->col
	 int mdim = c1blk.rows();
	 int vdim = c2blk.rows();
	 int rdim = blk[0].rows();
	 int cdim = blk[0].cols();
	 for(int mv=0; mv<mvdim; mv++){
            int m = mv/vdim;
	    int v = mv%vdim;
	    for(int c=0; c<cdim; c++){
	       for(int r=0; r<rdim; r++){
	          blk[m](r,c) = ecore + lblk(r,r) + c1blk(m,m) + c2blk(v,v) + rblk(c,c);
	       } // r
	    } // c
	 } // mv
      }
   } // qblocks
   vector<double> diag(wf.get_dim());
   wf.to_array(diag.data());
   return diag;
}


void tns::get_twodot_Hx(double* y,
		        const double* x,
		        const comb& icomb,
		        const comb_coord& p,
	                oper_dict& c1qops,
			oper_dict& c2qops,
		        oper_dict& lqops,
		        oper_dict& rqops,
		        const integral::two_body& int2e,
	                const integral::one_body& int1e,
		        const double ecore,
		        qtensor4& wf){
   bool debug = true;
   // const term
   wf.from_array(x);
   auto Hwf = ecore*wf;
   // construct H*wf
   int ifPl = lqops.find('P') != lqops.end();
   int ifPr = rqops.find('P') != rqops.end();
   assert(ifPl + ifPr == 1);
   if(debug) cout << "tns::get_twodot_Hx " << (ifPl? "PQ|AB" : "AB|PQ") << endl;
   // 1. H^LC1
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   auto Hwf1 = oper_kernel_Hwf("lc",wf1,lqops,c1qops,int2e,int1e); // Hwf1[l,c1,c2r]
   // 2. H^C2R
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   auto Hwf2 = oper_kernel_Hwf("cr",wf2,c2qops,rqops,int2e,int1e); // Hwf2[lc1,c2,r] 
   Hwf += Hwf2.split_lc1(wf.qrow,wf.qmid,wf.dpt_lc1().second);
   // 3. p1^LC1+*Sp1^C2R + h.c
   // p1^L+
   for(const auto& op1C : lqops['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      // p1^L1+*Sp1^C2R
      auto tmp2 = oper_kernel_Swf("cr",wf2,c2qops,rqops,int2e,int1e,p1); // tmp2[lc1,c2,r]
      auto tmp1 = tmp2.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_OIwf("lc",tmp1,op1); // tmp1[l,c1,c2r]
      // -Sp1^C2R+*p1^L1
      auto tmp2hc = oper_kernel_Swf("cr",wf2,c2qops,rqops,int2e,int1e,p1,true);
      auto tmp1hc = tmp2hc.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_OIwf("lc",tmp1hc,op1.T());
   }
   // p1^C1+
   for(const auto& op1C : c1qops['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      // p1^C1+*Sp1^C2R
      auto tmp2 = oper_kernel_Swf("cr",wf2,c2qops,rqops,int2e,int1e,p1);
      auto tmp1 = tmp2.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_IOwf("lc",tmp1,op1,1);
      // -Sp1^C2R+*p1^C1
      auto tmp2hc = oper_kernel_Swf("cr",wf2,c2qops,rqops,int2e,int1e,p1,true);
      auto tmp1hc = tmp2hc.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_IOwf("lc",tmp1hc,op1.T(),1);
   }
   // 4. q2^C2R+*Sq2^LC1 + h.c.
   // q2^C2+
   for(const auto& op2C : c2qops['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      // q2^C2+*Sq2^LC1 = -Sq2^LC1*q2^C2+
      auto tmp2 = oper_kernel_OIwf("cr",wf2,op2); // tmp2[lc1,c2,r]
      auto tmp1 = tmp2.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_Swf("lc",tmp1,lqops,c1qops,int2e,int1e,q2);
      // Sq2^LC1+*q2^C2
      auto tmp2hc = oper_kernel_OIwf("cr",wf2,op2.T());
      auto tmp1hc = tmp2hc.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_Swf("lc",tmp1hc,lqops,c1qops,int2e,int1e,q2,true);
   }
   // q2^R+
   for(const auto& op2C : rqops['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      // q2^R+*Sq2^LC1 = -Sq2^LC1*q2^R+
      auto tmp2 = oper_kernel_IOwf("cr",wf2,op2,1);
      auto tmp1 = tmp2.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_Swf("lc",tmp1,lqops,c1qops,int2e,int1e,q2);
      // Sq2^LC1+*q2^R
      auto tmp2hc = oper_kernel_IOwf("cr",wf2,op2.T(),1);
      auto tmp1hc = tmp2hc.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_Swf("lc",tmp1hc,lqops,c1qops,int2e,int1e,q2,true);
   }
   if(ifPl){   
      // 5. Ars^C2R*Prs^LC1 + h.c.
      // Ars^C2*Prs^LC1 = Prs^LC1*Ars^C2
      for(const auto& op2A : c2qops['A']){
	 int index = op2A.first;
	 const auto& op2 = op2A.second;
	 auto tmp2 = oper_kernel_OIwf("cr",wf2,op2,0);
         auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Pwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
	 auto tmp2hc = oper_kernel_OIwf("cr",wf2,op2.T(),0);
	 auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Pwf("lc",tmp1hc,lqops,c1qops,int2e,int1e,index,true);
      }
      // Ars^R*Prs^LC1 = Prs^LC1*Ars^R
      for(const auto& op2A : rqops['A']){
	 int index = op2A.first;
	 const auto& op2 = op2A.second;
	 auto tmp2 = oper_kernel_IOwf("cr",wf2,op2,0);
         auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Pwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
	 auto tmp2hc = oper_kernel_IOwf("cr",wf2,op2.T(),0);
	 auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Pwf("lc",tmp1hc,lqops,c1qops,int2e,int1e,index,true);
      }
      // Ars = r^C2+s^R+ (r<s)
      for(const auto& opCr : c2qops['C']){
         int r = opCr.first;
         for(const auto& opCs : rqops['C']){
	    int s = opCs.first;
	    if(r >= s) continue;
	    int index = oper_pack(r,s);
            // Ars = r^C2+s^R+ (r<s) 
	    auto tmp2 = oper_kernel_OOwf("cr",wf2,opCr.second,opCs.second,1);
            auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 += oper_kernel_Pwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
      	    // (r^C2+s^R+)^+ = s^R*r^C2 = -r^C2*s^R
      	    auto tmp2hc = oper_kernel_OOwf("cr",wf2,opCr.second.T(),opCs.second.T(),1);
      	    auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      	    Hwf1 -= oper_kernel_Pwf("lc",tmp1hc,lqops,c1qops,int2e,int1e,index,true);
	 }
      }
      // Ars = r^R+s^C2+ (r<s) 
      for(const auto& opCr : rqops['C']){
	 int r = opCr.first;
         for(const auto& opCs : c2qops['C']){
            int s = opCs.first;
	    if(r >= s) continue;
	    int index = oper_pack(r,s);
            // Ars = r^R+s^C2+ (r<s) = -s^C2+*r^R 
	    auto tmp2 = oper_kernel_OOwf("cr",wf2,opCs.second,opCr.second,1);
            auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 -= oper_kernel_Pwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
      	    // (r^R+s^C2+)^+ = s^C2*r^R 
      	    auto tmp2hc = oper_kernel_OOwf("cr",wf2,opCs.second.T(),opCr.second.T(),1);
      	    auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      	    Hwf1 += oper_kernel_Pwf("lc",tmp1hc,lqops,c1qops,int2e,int1e,index,true);
	 }
      }
      // 6. Qqr^LC1*Bqr^C2R
      // Bqr^C2
      for(const auto& op2B : c2qops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 auto tmp2 = oper_kernel_OIwf("cr",wf2,op2,0);
         auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Qwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
      }
      // Bqr^R
      for(const auto& op2B : rqops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 auto tmp2 = oper_kernel_IOwf("cr",wf2,op2,0);
         auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_Qwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
      }
      // Bqr = q^C2+*r^R
      for(const auto& opCq : c2qops['C']){
         int q = opCq.first;
         for(const auto& opCr : rqops['C']){
	    int r = opCr.first;
	    int index = oper_pack(q,r);
	    auto tmp2 = oper_kernel_OOwf("cr",wf2,opCq.second,opCr.second.T(),1);
            auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 += oper_kernel_Qwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
	 }
      }
      // Bqr = q^R+*r^C2 = -r^C2*q^R+
      for(const auto& opCq : rqops['C']){
	 int q = opCq.first;
         for(const auto& opCr : c2qops['C']){
            int r = opCr.first;
	    int index = oper_pack(q,r);
	    auto tmp2 = oper_kernel_OOwf("cr",wf2,opCr.second.T(),opCq.second,1);
            auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 -= oper_kernel_Qwf("lc",tmp1,lqops,c1qops,int2e,int1e,index);
         }
      }
   }else{
      // 5. Apq^LC1*Ppq^C2R + h.c.
      // Apq^L
      for(const auto& op1A : lqops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 auto tmp2 = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	 auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_OIwf("lc",tmp1,op1);
	 auto tmp2hc = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index,true);
	 auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_OIwf("lc",tmp1hc,op1.T());
      }
      // Apq^C1
      for(const auto& op1A : c1qops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 auto tmp2 = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	 auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_IOwf("lc",tmp1,op1,0);
	 auto tmp2hc = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index,true);
	 auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_IOwf("lc",tmp1hc,op1.T(),0);
      }
      // Apq = p^L+*q^C1+ (p<q)
      for(const auto& opCp : lqops['C']){
         int p = opCp.first;
	 for(const auto& opCq : c1qops['C']){
	    int q = opCq.first;
	    if(p >= q) continue;
	    int index = oper_pack(p,q);
	    // Apq = p^L+*q^C1+ (p<q)
	    auto tmp2 = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	    auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 += oper_kernel_OOwf("lc",tmp1,opCp.second,opCq.second,1); 
	    auto tmp2hc = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index,true);
	    auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 -= oper_kernel_OOwf("lc",tmp1hc,opCp.second.T(),opCq.second.T(),1); 
	 }	
      }
      // Apq = p^C1+*q^L+ (p<q) = -q^L+*p^C1+ 
      for(const auto& opCp : c1qops['C']){
         int p = opCp.first;
	 for(const auto& opCq : lqops['C']){
	    int q = opCq.first;
	    if(p >= q) continue;
	    int index = oper_pack(p,q);
	    // Apq = p^L+*q^C1+ (p<q)
	    auto tmp2 = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	    auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 -= oper_kernel_OOwf("lc",tmp1,opCq.second,opCp.second,1); 
	    auto tmp2hc = oper_kernel_Pwf("cr",wf2,c2qops,rqops,int2e,int1e,index,true);
	    auto tmp1hc = tmp2hc.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 += oper_kernel_OOwf("lc",tmp1hc,opCq.second.T(),opCp.second.T(),1); 
	 }
      }
      // 6. Bps^LC1*Qps^C2R
      // Bps^L
      for(const auto& op1B : lqops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 auto tmp2 = oper_kernel_Qwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	 auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_OIwf("lc",tmp1,op1);
      }
      // Bps^C1
      for(const auto& op1B : c1qops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 auto tmp2 = oper_kernel_Qwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	 auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_kernel_IOwf("lc",tmp1,op1,0);
      }
      // Bps = p^L+*s^C1 
      for(const auto& opCp : lqops['C']){
         int p = opCp.first;
	 for(const auto& opCs : c1qops['C']){
	    int s = opCs.first;
	    int index = oper_pack(p,s);
	    auto tmp2 = oper_kernel_Qwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	    auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 += oper_kernel_OOwf("lc",tmp1,opCp.second,opCs.second.T(),1); 
	 }
      }
      // Bps = p^C1+*s^L = -s^L*p^C1+ 
      for(const auto& opCp : c1qops['C']){
         int p = opCp.first;
	 for(const auto& opCs : lqops['C']){
	    int s = opCs.first;
	    int index = oper_pack(p,s);
	    auto tmp2 = oper_kernel_Qwf("cr",wf2,c2qops,rqops,int2e,int1e,index);
	    auto tmp1 = tmp2.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	    Hwf1 -= oper_kernel_OOwf("lc",tmp1,opCs.second.T(),opCp.second,1); 
	 }
      }
   }
   Hwf += Hwf1.split_c2r(wf.qver,wf.qcol,wf.dpt_c2r().second); 
   // finally copy back to y
   Hwf.to_array(y);
}
