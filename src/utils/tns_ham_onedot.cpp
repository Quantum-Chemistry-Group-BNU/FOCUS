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
   int ifB1Q2 = (lqops.find('B') != lqops.end() && rqops.find('Q') != rqops.end()); 
   int ifQ1B2 = (lqops.find('Q') != lqops.end() && rqops.find('B') != rqops.end());
   int ifB1B2 = (lqops.find('B') != lqops.end() && rqops.find('B') != rqops.end());
   assert(ifB1Q2 + ifQ1B2 + ifB1B2 == 1);
   if(ifB1Q2){
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
   }else if(ifQ1B2){
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
   }else if(ifB1B2){
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
	    // B^L---*---B^R
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
            // B^L*B^R : <p1q2||s1r2>p1q2r2s1 => <p1q2||s1r2>Bps^1*Bqr^2 
            for(auto& Bl : lqops['B']){
               if(Bl.second.sym != qsym(0,0)) continue;
               auto kps = oper_unpack(Bl.first);
               int kp = kps.first;
               int ks = kps.second;
               for(auto& Br : rqops['B']){
                  if(Br.second.sym != qsym(0,0)) continue;
                  auto kqr = oper_unpack(Br.first);
                  int kq = kqr.first;
                  int kr = kqr.second;
                  auto& lblk = Bl.second.qblocks[make_pair(qr,qr)];
                  auto& rblk = Br.second.qblocks[make_pair(qc,qc)];
                  for(int m=0; m<mdim; m++){
                     for(int c=0; c<cdim; c++){
                        for(int r=0; r<rdim; r++){
                           blk[m](r,c) += lblk(r,r)*rblk(c,c)*int2e.getAnti(kp,kq,ks,kr);
                        } // r
                     } // c
                  } // m
	       }
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
   int ifA1P2 = (lqops.find('A') != lqops.end() && rqops.find('P') != rqops.end()); 
   int ifP1A2 = (lqops.find('P') != lqops.end() && rqops.find('A') != rqops.end());
   int ifA1A2 = (lqops.find('A') != lqops.end() && rqops.find('A') != rqops.end());
   assert(ifA1P2 + ifP1A2 + ifA1A2 == 1);
   bool ifPl = ifP1A2 || ifA1A2;
   bool ifPr = ifA1P2;
   assert(ifPl + ifPr == 1);
   if(debug) cout << "tns::get_onedot_Hx " << (ifPl? "PQ|AB" : "AB|PQ") << endl;
   if(ifPl){
      // L=lc, R=r
      // 1. H^lc
      Hwf += oper_kernel_Hwf("lc",wf,lqops,cqops,int2e,int1e);
      // 2. H^r
      Hwf += contract_qt3_qt2_r(wf,rqops['H'][0]);
      // 3. p1^lc+*Sp1^r + h.c.
      // p1^l+*Sp1^r
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1); // special treatment of mid sign
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1.T(),op2S.T(),1);
      }
      // p1^c+*Sp1^r
      for(const auto& op1C : cqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 Hwf += oper_kernel_OOwf("cr",wf,op1,op2S,1);
	 Hwf -= oper_kernel_OOwf("cr",wf,op1.T(),op2S.T(),1);
      }
      // 4. q2^r+*Sq2^lc + h.c.
      for(const auto& op2C : rqops['C']){
         int q2 = op2C.first;
  	 const auto& op2 = op2C.second;
	 // q2^r+*Sq2^lc = -Sq2^lc*q2^r
	 auto qt3 = oper_kernel_IOwf("cr",wf,op2,1); 
	 Hwf -= oper_kernel_Swf("lc",qt3.row_signed(),lqops,cqops,int2e,int1e,q2);
	 auto qt3hc = oper_kernel_IOwf("cr",wf,op2.T(),1);
	 Hwf += oper_kernel_Swf("lc",qt3hc.row_signed(),lqops,cqops,int2e,int1e,q2,true);
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& op2A : rqops['A']){
         int index = op2A.first;
	 const auto& op2 = op2A.second;
	 // Ars^r*Prs^lc = Prs^lc*Ars^r
	 auto qt3 = oper_kernel_IOwf("cr",wf,op2,0);
	 Hwf += oper_kernel_Pwf("lc",qt3,lqops,cqops,int2e,int1e,index);
	 auto qt3hc = oper_kernel_IOwf("cr",wf,op2.T(),0);
	 Hwf += oper_kernel_Pwf("lc",qt3hc,lqops,cqops,int2e,int1e,index,true);
      }
      // 6. Qqr^lc*Bqr^r
      for(const auto& op2B : rqops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 auto qt3 = oper_kernel_IOwf("cr",wf,op2,0);
	 Hwf += oper_kernel_Qwf("lc",qt3,lqops,cqops,int2e,int1e,index);
      }
   }else{
      // L=l, R=cr 
      // 1. Hl 
      Hwf += contract_qt3_qt2_l(wf,lqops['H'][0]);
      // 2. Hcr
      Hwf += oper_kernel_Hwf("cr",wf,cqops,rqops,int2e,int1e);
      // 3. p1^l+*Sp1^cr + h.c.
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto op1 = op1C.second;
	 auto qt3 = oper_kernel_Swf("cr",wf,cqops,rqops,int2e,int1e,p1);
	 Hwf += oper_kernel_OIwf("lc",qt3.row_signed(),op1); // both lc/lr can work 
	 auto qt3hc = oper_kernel_Swf("cr",wf,cqops,rqops,int2e,int1e,p1,true); 
	 Hwf -= oper_kernel_OIwf("lc",qt3hc.row_signed(),op1.T());
      }
      // 4. q2^cr+*Sq2^l + h.c.
      // q2^c+*Sq2^l = -Sq2^l*q2^c+
      for(const auto& op2C : cqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lc",wf,op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lc",wf,op1S.T(),op2.T(),1);
      }
      // q2^r+*Sq2^l = -Sq2^l*q2^r+
      for(const auto& op2C : rqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1S.T(),op2.T(),1);
      }
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& op1A : lqops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 auto qt3 = oper_kernel_Pwf("cr",wf,cqops,rqops,int2e,int1e,index);
	 Hwf += oper_kernel_OIwf("lc",qt3,op1);
	 auto qt3hc = oper_kernel_Pwf("cr",wf,cqops,rqops,int2e,int1e,index,true);
	 Hwf += oper_kernel_OIwf("lc",qt3hc,op1.T());
      }
      // 6. Bps^l*Qps^cr
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
