#ifndef SWEEP_ONEDOT_HAM_H
#define SWEEP_ONEDOT_HAM_H

#include "oper_dict.h"
#include "oper_combine.h"

namespace ctns{
   
const bool debug_onedot_ham = false;
extern const bool debug_onedot_ham;

// local 
template <typename Tm>
void onedot_Hdiag_local(oper_dict<Tm>& cqops,
		        oper_dict<Tm>& lqops,
		        oper_dict<Tm>& rqops,
			const double ecore,
		        qtensor3<Tm>& wf){
   const auto& Hc = cqops('H')[0];
   const auto& Hl = lqops('H')[0];
   const auto& Hr = rqops('H')[0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(int bm=0; bm<wf.mids(); bm++){
      for(int br=0; br<wf.rows(); br++){
         for(int bc=0; bc<wf.cols(); bc++){
            auto& blk = wf(bm,br,bc);
	    if(blk.size() == 0) continue;
	    int mdim = wf.qmid.get_dim(bm);
	    int rdim = wf.qrow.get_dim(br);  
	    int cdim = wf.qcol.get_dim(bc);
	    // 1. local contributions: all four indices in c/l/r
	    const auto& cblk = Hc(bm,bm); // central->mid 
	    const auto& lblk = Hl(br,br); // left->row 
	    const auto& rblk = Hr(bc,bc); // row->col
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
}

// Ol*Oc*Ir
template <typename Tm>
void onedot_Hdiag_OlOc(const qtensor2<Tm>& Ol,
		       const qtensor2<Tm>& Oc,
		       qtensor3<Tm>& wf,
		       const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int br=0; br<wf.rows(); br++){
         for(int bc=0; bc<wf.cols(); bc++){
            auto& blk = wf(bm,br,bc);
            if(blk.size() == 0) continue;
            int mdim = wf.qmid.get_dim(bm);
            int cdim = wf.qcol.get_dim(bc);
            int rdim = wf.qrow.get_dim(br);  
            const auto& lblk = Ol(br,br);
            const auto& cblk = Oc(bm,bm);
            for(int m=0; m<mdim; m++){
               for(int c=0; c<cdim; c++){
                  for(int r=0; r<rdim; r++){
                     blk[m](r,c) += wt*lblk(r,r)*cblk(m,m);
                  } // r
               } // c
            } // m
         } // bc
      } // br
   } // bm
}

// Ol*Ic*Or
template <typename Tm>
void onedot_Hdiag_OlOr(const qtensor2<Tm>& Ol,
		       const qtensor2<Tm>& Or,
		       qtensor3<Tm>& wf,
		       const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int br=0; br<wf.rows(); br++){
         for(int bc=0; bc<wf.cols(); bc++){
            auto& blk = wf(bm,br,bc);
            if(blk.size() == 0) continue;
            int mdim = wf.qmid.get_dim(bm);
            int cdim = wf.qcol.get_dim(bc);
            int rdim = wf.qrow.get_dim(br);  
            const auto& lblk = Ol(br,br);
            const auto& rblk = Or(bc,bc);
            for(int m=0; m<mdim; m++){
               for(int c=0; c<cdim; c++){
                  for(int r=0; r<rdim; r++){
                     blk[m](r,c) += wt*lblk(r,r)*rblk(c,c);
                  } // r
               } // c
            } // m
         } // bc
      } // br
   } // bm
}

// Il*Oc*Or
template <typename Tm>
void onedot_Hdiag_OcOr(const qtensor2<Tm>& Oc,
		       const qtensor2<Tm>& Or,
		       qtensor3<Tm>& wf,
		       const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int br=0; br<wf.rows(); br++){
         for(int bc=0; bc<wf.cols(); bc++){
            auto& blk = wf(bm,br,bc);
            if(blk.size() == 0) continue;
            int mdim = wf.qmid.get_dim(bm);
            int cdim = wf.qcol.get_dim(bc);
            int rdim = wf.qrow.get_dim(br);  
            const auto& cblk = Oc(bm,bm);
            const auto& rblk = Or(bc,bc);
            for(int m=0; m<mdim; m++){
               for(int c=0; c<cdim; c++){
                  for(int r=0; r<rdim; r++){
                     blk[m](r,c) += wt*cblk(m,m)*rblk(c,c);
                  } // r
               } // c
            } // m
         } // bc
      } // br
   } // bm
}

template <typename Tm>
std::vector<double> onedot_Hdiag(const bool& ifkr,
	       			 const bool& ifNC,
				 oper_dict<Tm>& cqops,
			         oper_dict<Tm>& lqops,
			         oper_dict<Tm>& rqops,
			         const double ecore,
			         qtensor3<Tm>& wf,
	       	                 const int size,
	       	                 const int rank){
   if(debug_onedot_ham) std::cout << "ctns::onedot_Hdiag ifkr=" << ifkr << std::endl;
   //
   // 1. local terms: <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   //
   onedot_Hdiag_local(cqops, lqops, rqops, ecore/size, wf);
   //
   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //
   //         B^C
   //         |
   // B/Q^L---*---Q/B^R
   // 
   auto bindexC = oper_index_opB(cqops.cindex, ifkr);
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? lqops.cindex : rqops.cindex;
   auto bindex = oper_index_opB(cindex, ifkr); 
   if(not ifkr){
      // Q^L*B^C and B^C*Q^R
      for(const auto& index : bindexC){
         int iproc = distribute2(index,size);
 	 if(iproc == rank){
	    const auto& Oc = cqops('B').at(index);
	    if(Oc.sym != qsym()) continue; // screening for <c|B^C_{pq}|c>
	    const Tm wt = 2.0*wfac(index); // taking into account B^d*Q^d
	    const auto& Ol = lqops('Q').at(index);
	    onedot_Hdiag_OlOc(Ol,Oc,wf,wt);
	    const auto& Or = rqops('Q').at(index);
            onedot_Hdiag_OcOr(Oc,Or,wf,wt);
	 }
      } 
      // B^L*Q^R or Q^L*B^R 
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
 	 if(iproc == rank){
            const auto& Ol = lqops(BQ1).at(index);
	    if(Ol.sym != qsym()) continue; // screening for <l|B/Q^l_{pq}|l>
	    const Tm wt = 2.0*wfac(index); // taking into account B^d*Q^d
            const auto& Or = rqops(BQ2).at(index);
            onedot_Hdiag_OlOr(Ol,Or,wf,wt);
         } 
      }
   }else{
      // Q^L*B^C and B^C*Q^R
      for(const auto& index : bindexC){
         int iproc = distribute2(index,size);
 	 if(iproc == rank){
            const auto& Oc_A = cqops('B').at(index); // Baa/Bab[skipped]
            if(Oc_A.sym != qsym()) continue; // screening for <c|B^C_{pq}|c>
	    const Tm wt = 2.0*wfacBQ(index); 
	    const auto& Oc_B = Oc_A.K(0);
            const auto& Ol_A = lqops('Q').at(index);
	    const auto& Ol_B = Ol_A.K(0);
            onedot_Hdiag_OlOc(Ol_A,Oc_A,wf,wt);
            onedot_Hdiag_OlOc(Ol_B,Oc_B,wf,wt);
            const auto& Or_A = rqops('Q').at(index);
	    const auto& Or_B = Or_A.K(0);
            onedot_Hdiag_OcOr(Oc_A,Or_A,wf,wt);
            onedot_Hdiag_OcOr(Oc_B,Or_B,wf,wt);
	 }
      } 
      // B^L*Q^R or Q^L*B^R 
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
 	 if(iproc == rank){
            const auto& Ol_A = lqops(BQ1).at(index);
            if(Ol_A.sym != qsym()) continue;
	    const Tm wt = 2.0*wfacBQ(index);
	    const auto& Ol_B = Ol_A.K(0);
	    const auto& Or_A = rqops(BQ2).at(index);
	    const auto& Or_B = Or_A.K(0); 
            onedot_Hdiag_OlOr(Ol_A,Or_A,wf,wt);
            onedot_Hdiag_OlOr(Ol_B,Or_B,wf,wt);
         }
      }
   } // ifkr
   // save to real vector
   int ndim = wf.get_dim();
   std::vector<Tm> tmp(ndim);
   wf.to_array(tmp.data());
   // convert Tm to double
   std::vector<double> diag(ndim);
   std::transform(tmp.begin(), tmp.end(), diag.begin(),
                  [](const Tm& x){ return std::real(x); });
   return diag;
}

template <typename Tm> 
void onedot_Hx(Tm* y,
	       const Tm* x,
	       const int& isym,
	       const bool& ifkr,
	       const bool& ifNC,
	       oper_dict<Tm>& cqops,
	       oper_dict<Tm>& lqops,
	       oper_dict<Tm>& rqops,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore,
	       qtensor3<Tm>& wf,
	       const int size,
	       const int rank){
   if(debug_onedot_ham) std::cout << "ctns::onedot_Hx ifkr=" << ifkr << std::endl;
   const bool dagger = true;
   const Tm scale = ifkr? 0.5 : 1.0;
   //
   // constant term
   //
   wf.from_array(x);
   const Tm fac = scale*(ecore/size);
   qtensor3<Tm> Hwf = fac*wf;
   //
   // construct H*wf: if ifkr=True, construct skeleton sigma vector 
   //
   if(ifNC){
      // L=l, R=cr: Al*Pr+Bl*Qr 
      // 1. H^l 
      Hwf += scale*contract_qt3_qt2_l(wf,lqops('H')[0]);
      // 2. H^cr
      Hwf += scale*oper_compxwf_opH("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,size,rank);
      // 3. p1^l+*Sp1^cr + h.c.
      //    ol*or|lcr>psi[lcr] => ol|l>*or|cr>(-1)^{p(l)}psi[lcr]
      for(const auto& op1C : lqops('C')){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 auto qt3n = oper_compxwf_opS("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,p1,size,rank);
	 auto qt3h = oper_compxwf_opS("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,p1,size,rank,dagger); 
	 Hwf += oper_kernel_OIwf("lc",qt3n.row_signed(),op1); // both lc/lr can work 
	 Hwf -= oper_kernel_OIwf("lc",qt3h.row_signed(),op1,dagger);
      }

      // 4. q2^cr+*Sq2^l + h.c.
      auto infoC = oper_combine_opC(cqops.cindex, rqops.cindex);
      for(const auto pr : infoC){
         int iformula = pr.first;
         int index = pr.second;
	 const auto& op1 = lqops('S').at(index);
         auto qt3n = oper_normxwf_opC("cr",wf,cqops,rqops,iformula,index);
	 auto qt3h = oper_normxwf_opC("cr",wf,cqops,rqops,iformula,index,dagger);
	 Hwf -= oper_kernel_OIwf("lc",qt3n.row_signed(),op1); 
	 Hwf += oper_kernel_OIwf("lc",qt3h.row_signed(),op1,dagger); 
      }

/*
      // 4.1 q2^c+*Sq2^l = -Sq2^l*q2^c+
      for(const auto& op2C : cqops('C')){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops('S').at(q2);
	 Hwf -= oper_kernel_OOwf("lc",wf,op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lc",wf,op1S,op2,1,dagger);
      }
      // 4.2 q2^r+*Sq2^l = -Sq2^l*q2^r+
      for(const auto& op2C : rqops('C')){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops('S').at(q2);
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1,dagger);
      }
*/     
      // 5. Apq^l*Ppq^cr + h.c.
      auto aindex = oper_index_opA(lqops.cindex, ifkr);
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacAP(index) : 1.0;
            const auto& op1 = lqops('A').at(index);
	    auto qt3n = oper_compxwf_opP("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index);
	    auto qt3h = oper_compxwf_opP("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	    Hwf += wt*oper_kernel_OIwf("lc",qt3n,op1);
	    Hwf += wt*oper_kernel_OIwf("lc",qt3h,op1,dagger);
	 }
      }
      // 6. Bps^l*Qps^cr (using Hermicity)
      auto bindex = oper_index_opB(lqops.cindex, ifkr);
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacBQ(index) : wfac(index);
            const auto& op1 = lqops('B').at(index);
	    auto qt3n = oper_compxwf_opQ("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index);
	    auto qt3h = oper_compxwf_opQ("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	    Hwf += wt*oper_kernel_OIwf("lc",qt3n,op1);
	    Hwf += wt*oper_kernel_OIwf("lc",qt3h,op1,dagger);
	 }
      }
   // Ar*Pl+Br*Ql => L=lc, R=r
   }else{
/*
      // 1. H^lc
      Hwf += scale*oper_compxwf_opH("lc",wf,lqops,cqops,isym,ifkr,int2e,int1e,size,rank);
      // 2. H^r
      Hwf += scale*contract_qt3_qt2_r(wf,rqops('H')[0]);
      // 3. p1^lc+*Sp1^r + h.c.
      // 3.1 p1^l+*Sp1^r
      for(const auto& op1C : lqops('C')){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops('S').at(p1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1); // special treatment of mid sign
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1,dagger);
      }
      // 3.2 p1^c+*Sp1^r
      for(const auto& op1C : cqops('C')){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops('S').at(p1);
	 Hwf += oper_kernel_OOwf("cr",wf,op1,op2S,1);
	 Hwf -= oper_kernel_OOwf("cr",wf,op1,op2S,1,dagger);
      }
      // 4. q2^r+*Sq2^lc + h.c.
      for(const auto& op2C : rqops('C')){
         int q2 = op2C.first;
  	 const auto& op2 = op2C.second;
	 // q2^r+*Sq2^lc = -Sq2^lc*q2^r
	 auto qt3n = oper_kernel_IOwf("cr",wf,op2,1); 
	 auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
	 Hwf -= oper_compxwf_opS("lc",qt3n.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2,size,rank);
	 Hwf += oper_compxwf_opS("lc",qt3h.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2,size,rank,dagger);
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& op2A : rqops('A')){
         int index = op2A.first;
	 const auto& op2 = op2A.second;
	 // Ars^r*Prs^lc = Prs^lc*Ars^r
	 const Tm wt = ifkr? wfacAP(index) : 1.0;
	 auto qt3n = wt*oper_kernel_IOwf("cr",wf,op2,0);
	 auto qt3h = wt*oper_kernel_IOwf("cr",wf,op2,0,dagger);
	 Hwf += oper_compxwf_opP("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	 Hwf += oper_compxwf_opP("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      for(const auto& op2B : rqops('B')){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	 auto qt3n = wt*oper_kernel_IOwf("cr",wf,op2,0);
	 auto qt3h = wt*oper_kernel_IOwf("cr",wf,op2,0,dagger);
	 Hwf += oper_compxwf_opQ("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	 Hwf += oper_compxwf_opQ("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
      }
*/
   } // ifNC
   Hwf.to_array(y);
}

} // ctns

#endif
