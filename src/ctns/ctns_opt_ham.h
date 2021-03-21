#ifndef CTNS_OPT_HAM_H
#define CTNS_OPT_HAM_H

#include "../core/integral.h"
#include "ctns_qtensor.h"
#include "ctns_comb.h"
#include "ctns_oper_util.h"

namespace ctns{

// Hdiag for one-dot case
template <typename Tm>
std::vector<double> get_onedot_Hdiag(oper_dict<Tm>& cqops,
			             oper_dict<Tm>& lqops,
			             oper_dict<Tm>& rqops,
			             const double ecore,
			             qtensor3<Tm>& wf){
   const bool debug = false;
   if(debug) std::cout << "\nctns::get_onedot_Hdiag" << std::endl;
/*   
   // 1. local contributions
   auto& Hc = cqops['H'][0];
   auto& Hl = lqops['H'][0];
   auto& Hr = rqops['H'][0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(int bm=0; bm<wf.mids(); bm++){
      auto& cblk = Hc(bm,bm); // central->mid
      int mdim = wf.qmid.get_dim(bm);
      for(int br=0; br<wf.rows(); br++){
         auto& lblk = Hl(br,br); // left->row
	 int rdim = wf.qrow.get_dim(br);
	 for(int bc=0; bc<wf.cols(); bc++){
	    auto& rblk = Hr(bc,bc); // row->col
	    int cdim = wf.qcol.get_dim(bc);
	    auto& blk = wf(bm,br,bc);
	    if(blk.size() == 0) continue;
	    for(int m=0; m<mdim; m++){
	       for(int c=0; c<cdim; c++){
	          for(int r=0; r<rdim; r++){
	             blk[m](r,c) = ecore + lblk(r,r) + cblk(m,m) + rblk(c,c);
	          } // r
	       } // c
	    } // m
	 } // bc
      } // br
   } // bm
   
   // 2. B*Q term - density-density interactions
   for(int bm=0; bm<wf.mids(); bm++){
      int mdim = wf.qmid.get_dim(bm);
      for(int br=0; br<wf.rows(); br++){
	 int rdim = wf.qrow.get_dim(br);
	 for(int bc=0; bc<wf.cols(); bc++){
	    int cdim = wf.qcol.get_dim(bc);
	    auto& blk = wf(bm,br,bc);
	    if(blk.size() == 0) continue;
            
	    // Q^L*B^C and B^C*Q^R
            for(auto& Bc : cqops['B']){
               if(Bc.second.sym != qsym(0,0)) continue; // <c|B^c|c>
	       auto& cblk = Bc.second(bm,bm);
	       // Q^L*B^C
	       auto& Ql = lqops['Q'].at(Bc.first);
	       auto& lblk = Ql(br,br);
               // B^C*Q^R
	       auto& Qr = rqops['Q'].at(Bc.first);
               auto& rblk = Qr(qc,qc);
               for(int m=0; m<mdim; m++){
                  for(int c=0; c<cdim; c++){
                     for(int r=0; r<rdim; r++){
                        blk[m](r,c) += (lblk(r,r)+rblk(c,c))*cblk(m,m);
                     } // r
                  } // c
               } // m
            } // BQ
            
	    // B^L*Q^R or Q^L*B^R 
            if(lqops['B'].size() < rqops['B'].size()){
	       for(auto& Bl : lqops['B']){
                  if(Bl.second.sym != qsym(0,0)) continue;
                  auto& lblk = Bl.second(br,br);
                  auto& Qr = rqops['Q'].at(Bl.first);
                  auto& rblk = Qr(bc,bc);
                  for(int m=0; m<mdim; m++){
                     for(int c=0; c<cdim; c++){
                        for(int r=0; r<rdim; r++){
                           blk[m](r,c) += lblk(r,r)*rblk(c,c);
                        } // r
                     } // c
                  } // m
               } // BQ
	    }else{
               for(auto& Br : rqops['B']){
                  if(Br.second.sym != qsym(0,0)) continue;
	          auto& rblk = Br.second(bc,bc);
	          auto& Ql = lqops['Q'].at(Br.first);
	          auto& lblk = Ql(br,br);
                  for(int m=0; m<mdim; m++){
                     for(int c=0; c<cdim; c++){
                        for(int r=0; r<rdim; r++){
                           blk[m](r,c) += lblk(r,r)*rblk(c,c);
                        } // r
                     } // c
                  } // m
               } // BQ
	    }

	 } // bc
      } // br
   } // bm
*/
   // save
   std::vector<double> diag(wf.get_dim());
/*
   wf.to_array(diag.data());
*/
   return diag;
}

/*
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
   bool debug = false;
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
*/

/*
void get_onedot_Hx(double* y,
	    	   const double* x,
	    	   const comb& icomb,
 	    	   const comb_coord& p,
	    	   oper_dict& cqops,
	    	   oper_dict& lqops,
	    	   oper_dict& rqops,
	    	   const integral::two_body& int2e,
	    	   const integral::one_body& int1e,
	    	   const double ecore,
	    	   qtensor3& wf);

// two-dot
std::vector<double> get_twodot_Hdiag(oper_dict& c1qops,
			             oper_dict& c2qops,
			             oper_dict& lqops,
			             oper_dict& rqops,
		        	     const integral::two_body& int2e,
			             const double ecore,
			             qtensor4& wf);

void get_twodot_Hx(double* y,
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
	    	   qtensor4& wf);
*/

} // ctns

#endif
