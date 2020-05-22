#include "tns_ham.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

vector<double> tns::get_onedot_Hdiag(oper_dict& cqops,
			             oper_dict& lqops,
			             oper_dict& rqops,
		        	     const integral::two_body& int2e,
			             const double ecore,
			             qtensor3& wf){
   cout << "\ntns::get_onedot_Hdiag" << endl;
   // 1. local contributions
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
	 int rdim = blk[0].rows();
	 int cdim = blk[0].cols();
	 auto& cblk = Hc.qblocks[make_pair(qm,qm)]; // central->mid
	 auto& lblk = Hl.qblocks[make_pair(qr,qr)]; // left->row
	 auto& rblk = Hr.qblocks[make_pair(qc,qc)]; // row->col
	 for(int m=0; m<mdim; m++){
	    for(int c=0; c<cdim; c++){
	       for(int r=0; r<rdim; r++){
	          blk[m](r,c) = ecore + lblk(r,r) + cblk(m,m) + rblk(c,c);
	       } // r
	    } // c
	 } // m
      }
   } // qblocks
   // 2. B*Q term - density-density interactions
   int ifPl = lqops.find('P') != lqops.end();
   int ifPr = rqops.find('P') != rqops.end();
   assert(ifPl + ifPr == 1);
   if(ifPl){
      for(auto& p : wf.qblocks){
         auto& blk = p.second;
         int mdim = blk.size();
         if(mdim > 0){
            auto& q = p.first;
            auto qm = get<0>(q);
            auto qr = get<1>(q);
            auto qc = get<2>(q);
            int rdim = blk[0].rows();
            int cdim = blk[0].cols();
	    //
	    //      B^C
	    //       |
	    // Q^L---*---B^R
	    //
            // Q^L*B^C
            for(auto& Bc : cqops['B']){
               if(Bc.second.sym != qsym(0,0)) continue;
	       auto& Ql = lqops['Q'].at(Bc.first);
	       auto& lblk = Ql.qblocks[make_pair(qr,qr)];
	       auto& cblk = Bc.second.qblocks[make_pair(qm,qm)];
               for(int m=0; m<mdim; m++){
                  for(int c=0; c<cdim; c++){
                     for(int r=0; r<rdim; r++){
                        blk[m](r,c) += lblk(r,r)*cblk(m,m);
                     } // r
                  } // c
               } // m
            }
            // Q^L*B^R 
            for(auto& Br : rqops['B']){
               if(Br.second.sym != qsym(0,0)) continue;
	       auto& Ql = lqops['Q'].at(Br.first);
	       auto& lblk = Ql.qblocks[make_pair(qr,qr)];
	       auto& rblk = Br.second.qblocks[make_pair(qc,qc)];
               for(int m=0; m<mdim; m++){
                  for(int c=0; c<cdim; c++){
                     for(int r=0; r<rdim; r++){
                        blk[m](r,c) += lblk(r,r)*rblk(c,c);
                     } // r
                  } // c
               } // m
            } 
            // B^C*B^R : <p1q2||s1r2>p1q2r2s1 => <p1q2||s1r2>Bps^1*Bqr^2 
            for(auto& Bc : cqops['B']){
               if(Bc.second.sym != qsym(0,0)) continue;
               auto kps = oper_unpack(Bc.first);
               int kp = kps.first;
               int ks = kps.second;
               for(auto& Br : rqops['B']){
                  if(Br.second.sym != qsym(0,0)) continue;
                  auto kqr = oper_unpack(Br.first);
                  int kq = kqr.first;
                  int kr = kqr.second;
                  // <p1q2||r1s2>Bpr^1*Bqs^2
                  auto& cblk = Bc.second.qblocks[make_pair(qm,qm)]; // central->mid
                  auto& rblk = Br.second.qblocks[make_pair(qc,qc)]; // row->col
                  for(int m=0; m<mdim; m++){
                     for(int c=0; c<cdim; c++){
                        for(int r=0; r<rdim; r++){
                           blk[m](r,c) += cblk(m,m)*rblk(c,c)*int2e.getAnti(kp,kq,ks,kr);
                        } // r
                     } // c
                  } // m
	       }
	    }
         } // mdim
      } // qblocks
   }else{
      for(auto& p : wf.qblocks){
         auto& blk = p.second;
         int mdim = blk.size();
         if(mdim > 0){
            auto& q = p.first;
            auto qm = get<0>(q);
            auto qr = get<1>(q);
            auto qc = get<2>(q);
            int rdim = blk[0].rows();
            int cdim = blk[0].cols();
	    //
	    //      B^C
	    //       |
	    // B^L---*---Q^R
	    //
            // B^L*B^C : <p1q2||s1r2>p1q2r2s1 => <p1q2||s1r2>Bps^1*Bqr^2 
            for(auto& Bl : lqops['B']){
               if(Bl.second.sym != qsym(0,0)) continue;
               auto kps = oper_unpack(Bl.first);
               int kp = kps.first;
               int ks = kps.second;
               for(auto& Bc : cqops['B']){
                  if(Bc.second.sym != qsym(0,0)) continue;
                  auto kqr = oper_unpack(Bc.first);
                  int kq = kqr.first;
                  int kr = kqr.second;
                  auto& lblk = Bl.second.qblocks[make_pair(qr,qr)];
                  auto& cblk = Bc.second.qblocks[make_pair(qm,qm)];
                  for(int m=0; m<mdim; m++){
                     for(int c=0; c<cdim; c++){
                        for(int r=0; r<rdim; r++){
                           blk[m](r,c) += lblk(r,r)*cblk(m,m)*int2e.getAnti(kp,kq,ks,kr);
                        } // r
                     } // c
                  } // m
	       }
            }
	    // B^L*Q^R 
            for(auto& Bl : lqops['B']){
	       if(Bl.second.sym != qsym(0,0)) continue;
	       auto& Qr = rqops['Q'].at(Bl.first);
	       auto& lblk = Bl.second.qblocks[make_pair(qr,qr)];
	       auto& rblk = Qr.qblocks[make_pair(qc,qc)];
               for(int m=0; m<mdim; m++){
                  for(int c=0; c<cdim; c++){
                     for(int r=0; r<rdim; r++){
                        blk[m](r,c) += lblk(r,r)*rblk(c,c);
                     } // r
                  } // c
               } // m
            }
	    // B^C*Q^R 
            for(auto& Bc : cqops['B']){
	       if(Bc.second.sym != qsym(0,0)) continue;
	       auto& Qr = rqops['Q'].at(Bc.first);
	       auto& cblk = Bc.second.qblocks[make_pair(qm,qm)];
	       auto& rblk = Qr.qblocks[make_pair(qc,qc)];
               for(int m=0; m<mdim; m++){
                  for(int c=0; c<cdim; c++){
                     for(int r=0; r<rdim; r++){
                        blk[m](r,c) += cblk(m,m)*rblk(c,c);
                     } // r
                  } // c
               } // m
            }
         } // mdim
      } // qblocks
   }
   // save
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
   // const term
   wf.from_array(x);
   auto Hwf = ecore*wf;
   // construct H*wf
   int ifPl = lqops.find('P') != lqops.end();
   int ifPr = rqops.find('P') != rqops.end();
   assert(ifPl + ifPr == 1);
   if(debug) cout << "tns::get_onedot_Hx " << (ifPl? "LC|R" : "L|CR") << endl;
   if(ifPl){
      // L=lc, R=r
      // H1
      Hwf += oper_kernel_Hwf("lc",wf,lqops,cqops,int2e,int1e);
      // H2
      Hwf += contract_qt3_qt2_r(wf,rqops['H'][0]);
      // p1^+*Sp1^2 + h.c.
      for(const auto& op1C : cqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 Hwf += oper_kernel_OOwf("cr",wf,op1,op2S,1);
	 Hwf -= oper_kernel_OOwf("cr",wf,op1.T(),op2S.T(),1);
      }
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 // special treatment of sign
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1);
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1.T(),op2S.T(),1);
      }
      // q2^+*Sq2^1 + h.c.
      for(const auto& op2C : rqops['C']){
         int q2 = op2C.first;
  	 const auto& op2 = op2C.second;
         auto qt3 = oper_kernel_Swf("lc",wf,lqops,cqops,int2e,int1e,q2);
	 auto qt3hc = oper_kernel_Swf("lc",wf,lqops,cqops,int2e,int1e,q2,true);
	 // special treatment of sign
	 Hwf -= oper_kernel_IOwf("cr",qt3.row_signed(),op2,1);
	 Hwf += oper_kernel_IOwf("cr",qt3hc.row_signed(),op2.T(),1); 
      }
      // Ars^2*Prs^1 + h.c.
      for(const auto& op2A : rqops['A']){
         int index = op2A.first;
	 const auto& op2 = op2A.second;
	 auto qt3 = oper_kernel_Pwf("lc",wf,lqops,cqops,int2e,int1e,index);
	 auto qt3hc = oper_kernel_Pwf("lc",wf,lqops,cqops,int2e,int1e,index,true);
	 Hwf += oper_kernel_IOwf("cr",qt3,op2,0);
	 Hwf += oper_kernel_IOwf("cr",qt3hc,op2.T(),0);
      }
      // Qqr^1*Bqr^2
      for(const auto& op2B : rqops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 auto qt3 = oper_kernel_Qwf("lc",wf,lqops,cqops,int2e,int1e,index);
	 Hwf += oper_kernel_IOwf("cr",qt3,op2,0);
      }
   }else{
      // L=l, R=cr 
      // H1 
      Hwf += contract_qt3_qt2_l(wf,lqops['H'][0]);
      // H2
      Hwf += oper_kernel_Hwf("cr",wf,cqops,rqops,int2e,int1e);
      // p1^+*Sp1^2 + h.c.
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto op1 = op1C.second;
	 auto qt3 = oper_kernel_Swf("cr",wf,cqops,rqops,int2e,int1e,p1);
	 auto qt3hc = oper_kernel_Swf("cr",wf,cqops,rqops,int2e,int1e,p1,true); 
	 Hwf += oper_kernel_OIwf("lc",qt3.row_signed(),op1); // both lc/lr work here
	 Hwf -= oper_kernel_OIwf("lc",qt3hc.row_signed(),op1.T());
      }
      // q2^+*Sq2^1 + h.c.
      for(const auto& op2C : cqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lc",wf,op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lc",wf,op1S.T(),op2.T(),1);
      }
      for(const auto& op2C : rqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1S.T(),op2.T(),1);
      }
      // Apq^1*Ppq^2 + h.c.
      for(const auto& op1A : lqops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 auto qt3 = oper_kernel_Pwf("cr",wf,cqops,rqops,int2e,int1e,index);
	 auto qt3hc = oper_kernel_Pwf("cr",wf,cqops,rqops,int2e,int1e,index,true);
	 Hwf += oper_kernel_OIwf("lc",qt3,op1);
	 Hwf += oper_kernel_OIwf("lc",qt3hc,op1.T());
      }
      // Bps^1*Qps^2
      for(const auto& op1B : lqops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 auto qt3 = oper_kernel_Qwf("cr",wf,cqops,rqops,int2e,int1e,index);
	 Hwf += oper_kernel_OIwf("lc",qt3,op1);
      }
   }
   // finally copy back to y
   Hwf.to_array(y);
}
