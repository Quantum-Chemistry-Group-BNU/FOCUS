#ifndef SWEEP_ONEDOT_H
#define SWEEP_ONEDOT_H

namespace ctns{

// local 
template <typename Tm>
void onedot_Hdiag_local(oper_dict<Tm>& cqops,
		        oper_dict<Tm>& lqops,
		        oper_dict<Tm>& rqops,
		        const double ecore,
		        qtensor3<Tm>& wf){
   const auto& Hc = cqops['H'][0];
   const auto& Hl = lqops['H'][0];
   const auto& Hr = rqops['H'][0];
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   for(int bm=0; bm<wf.mids(); bm++){
      for(int br=0; br<wf.rows(); br++){
         for(int bc=0; bc<wf.cols(); bc++){
            auto& blk = wf(bm,br,bc);
	    if(blk.size() == 0) continue;
	    int mdim = wf.qmid.get_dim(bm);
	    int cdim = wf.qcol.get_dim(bc);
	    int rdim = wf.qrow.get_dim(br);  
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

template <typename Tm>
std::vector<double> onedot_Hdiag(const bool ifkr,
				 oper_dict<Tm>& cqops,
			         oper_dict<Tm>& lqops,
			         oper_dict<Tm>& rqops,
			         const double ecore,
			         qtensor3<Tm>& wf){
   bool debug = true;
   if(debug) std::cout << "\nctns::onedot_Hdiag ifkr=" << ifkr << std::endl;
   // <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   onedot_Hdiag_local(cqops, lqops, rqops, ecore, wf);
   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //         B^C
   //         |
   // B/Q^L---*---Q/B^R
   if(not ifkr){
      // B^C*Q^R 
      for(const auto& p : cqops['B']){
         const auto& Bc = p.second;
         if(Bc.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr = rqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         onedot_Hdiag_OcOr(Bc,Qr,wf,wt);
      } // op
      // Q^L*B^C 
      for(const auto& p : cqops['B']){
         const auto& Bc = p.second;
         if(Bc.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql = lqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         onedot_Hdiag_OlOc(Ql,Bc,wf,wt);
      } // op
      // B^L*Q^R or Q^L*B^R 
      if(lqops['B'].size() <= rqops['B'].size()){
         for(const auto& p : lqops['B']){
            const auto& Bl = p.second;
            if(Bl.sym != qsym()) continue;
            const auto& Qr = rqops['Q'].at(p.first);
	    const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
            onedot_Hdiag_OlOr(Bl,Qr,wf,wt);
         } // op
      }else{
         for(const auto& p: rqops['B']){
            const auto& Br = p.second;
            if(Br.sym != qsym()) continue;
            const auto& Ql = lqops['Q'].at(p.first);
	    const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
            onedot_Hdiag_OlOr(Ql,Br,wf,wt);
         } // op
      }
   }else{
      // B^C*Q^R 
      for(const auto& p : cqops['B']){
         const auto& Bc_A = p.second;
         if(Bc_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr_A = rqops['Q'].at(p.first);
	 const auto& Bc_B = Bc_A.K(0);
	 const auto& Qr_B = Qr_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); 
         onedot_Hdiag_OcOr(Bc_A,Qr_A,wf,wt);
         onedot_Hdiag_OcOr(Bc_B,Qr_B,wf,wt);
      } // op
      // Q^L*B^C 
      for(const auto& p : cqops['B']){
         const auto& Bc_A = p.second;
         if(Bc_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql_A = lqops['Q'].at(p.first);
	 const auto& Bc_B = Bc_A.K(0);
	 const auto& Ql_B = Ql_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); 
         onedot_Hdiag_OlOc(Ql_A,Bc_A,wf,wt);
         onedot_Hdiag_OlOc(Ql_B,Bc_B,wf,wt);
      } // op
      // B^L*Q^R or Q^L*B^R 
      if(lqops['B'].size() <= rqops['B'].size()){
         for(const auto& p : lqops['B']){
            const auto& Bl_A = p.second;
            if(Bl_A.sym != qsym()) continue;
            const auto& Qr_A = rqops['Q'].at(p.first);
	    const auto& Bl_B = Bl_A.K(0);
	    const auto& Qr_B = Qr_A.K(0); 
	    const Tm wt = 2.0*wfacBQ(p.first); 
            onedot_Hdiag_OlOr(Bl_A,Qr_A,wf,wt);
            onedot_Hdiag_OlOr(Bl_B,Qr_B,wf,wt);
         } // op
      }else{
         for(const auto& p: rqops['B']){
            const auto& Br_A = p.second;
            if(Br_A.sym != qsym()) continue;
            const auto& Ql_A = lqops['Q'].at(p.first);
	    const auto& Br_B = Br_A.K(0);
	    const auto& Ql_B = Ql_A.K(0);
	    const Tm wt = 2.0*wfacBQ(p.first); 
            onedot_Hdiag_OlOr(Ql_A,Br_A,wf,wt);
            onedot_Hdiag_OlOr(Ql_B,Br_B,wf,wt);
         } // op
      }
   }
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
	       oper_dict<Tm>& cqops,
	       oper_dict<Tm>& lqops,
	       oper_dict<Tm>& rqops,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore,
	       qtensor3<Tm>& wf){
   const bool debug = true;
   const bool dagger = true;
   if(debug) std::cout << "ctns::onedot_Hx ifkr=" << ifkr << std::endl;
   // const term
   wf.from_array(x);
   auto Hwf = ecore*wf;
   // construct H*wf
   int nl_opA = (lqops.find('A') != lqops.end())? lqops['A'].size() : 0;
   int nl_opB = (lqops.find('B') != lqops.end())? lqops['B'].size() : 0;
   int nr_opA = (rqops.find('A') != rqops.end())? rqops['A'].size() : 0;
   int nr_opB = (rqops.find('B') != rqops.end())? rqops['B'].size() : 0;
   if(debug){
      std::cout << "(nl_opA,nl_opB)=" << nl_opA << "," << nl_opB << " "
	        << "(nr_opA,nr_opB)=" << nr_opA << "," << nr_opB << std::endl; 
   }
   bool ifMergeCR = (nl_opA + nl_opB <= nr_opA + nr_opB)? true : false;

   ifMergeCR = false;

   // Al*Pr+Bl*Qr => L=l, R=cr
   if(ifMergeCR){

      // 1. H^l 
      Hwf += contract_qt3_qt2_l(wf,lqops['H'][0]);
      // 2. H^cr
      Hwf += oper_compxwf_opH("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e);
      // 3. p1^l+*Sp1^cr + h.c.
      //    ol*or|lcr>psi[lcr] => ol|l>*or|cr>(-1)^{p(l)}psi[lcr]
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto op1 = op1C.second;
	 auto qt3n = oper_compxwf_opS("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,p1);
	 auto qt3h = oper_compxwf_opS("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,p1,dagger); 
	 Hwf += oper_kernel_OIwf("lc",qt3n.row_signed(),op1); // both lc/lr can work 
	 Hwf -= oper_kernel_OIwf("lc",qt3h.row_signed(),op1,dagger);
      }
      // 4. q2^cr+*Sq2^l + h.c.
      // 4.1 q2^c+*Sq2^l = -Sq2^l*q2^c+
      for(const auto& op2C : cqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lc",wf,op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lc",wf,op1S,op2,1,dagger);
      }
      // 4.2 q2^r+*Sq2^l = -Sq2^l*q2^r+
      for(const auto& op2C : rqops['C']){
	 int q2 = op2C.first;
	 const auto& op2 = op2C.second;
	 const auto& op1S = lqops['S'].at(q2);
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1S,op2,1,dagger);
      }
      // 5. Apq^l*Ppq^cr + h.c.
      for(const auto& op1A : lqops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 auto qt3n = oper_compxwf_opP("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index);
	 auto qt3h = oper_compxwf_opP("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 Hwf += oper_kernel_OIwf("lc",qt3n,op1);
	 Hwf += oper_kernel_OIwf("lc",qt3h,op1,dagger);
      }
      // 6. Bps^l*Qps^cr (using Hermicity)
      for(const auto& op1B : lqops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 const Tm wt = wfac(index);
	 auto qt3n = oper_compxwf_opQ("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index);
	 auto qt3h = oper_compxwf_opQ("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 Hwf += wt*oper_kernel_OIwf("lc",qt3n,op1);
	 Hwf += wt*oper_kernel_OIwf("lc",qt3h,op1,dagger);
      }

   // Ar*Pl+Br*Ql => L=lc, R=r
   }else{

      // 1. H^lc
      Hwf += oper_compxwf_opH("lc",wf,lqops,cqops,isym,ifkr,int2e,int1e);
      // 2. H^r
      Hwf += contract_qt3_qt2_r(wf,rqops['H'][0]);
      // 3. p1^lc+*Sp1^r + h.c.
      // 3.1 p1^l+*Sp1^r
      for(const auto& op1C : lqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 Hwf += oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1); // special treatment of mid sign
	 Hwf -= oper_kernel_OOwf("lr",wf.mid_signed(),op1,op2S,1,dagger);
      }
      // 3.2 p1^c+*Sp1^r
      for(const auto& op1C : cqops['C']){
	 int p1 = op1C.first;
	 const auto& op1 = op1C.second;
	 const auto& op2S = rqops['S'].at(p1);
	 Hwf += oper_kernel_OOwf("cr",wf,op1,op2S,1);
	 Hwf -= oper_kernel_OOwf("cr",wf,op1,op2S,1,dagger);
      }
      // 4. q2^r+*Sq2^lc + h.c.
      for(const auto& op2C : rqops['C']){
         int q2 = op2C.first;
  	 const auto& op2 = op2C.second;
	 // q2^r+*Sq2^lc = -Sq2^lc*q2^r
	 auto qt3n = oper_kernel_IOwf("cr",wf,op2,1); 
	 auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
	 Hwf -= oper_compxwf_opS("lc",qt3n.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2);
	 Hwf += oper_compxwf_opS("lc",qt3h.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2,dagger);
      }
      // 5. Ars^r*Prs^lc + h.c.
      for(const auto& op2A : rqops['A']){
         int index = op2A.first;
	 const auto& op2 = op2A.second;
	 // Ars^r*Prs^lc = Prs^lc*Ars^r
	 auto qt3n = oper_kernel_IOwf("cr",wf,op2,0);
	 auto qt3h = oper_kernel_IOwf("cr",wf,op2,0,dagger);
	 Hwf += oper_compxwf_opP("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	 Hwf += oper_compxwf_opP("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      for(const auto& op2B : rqops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 const Tm wt = wfac(index);
	 auto qt3n = wt*oper_kernel_IOwf("cr",wf,op2,0);
	 auto qt3h = wt*oper_kernel_IOwf("cr",wf,op2,0,dagger);
	 Hwf += oper_compxwf_opQ("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	 Hwf += oper_compxwf_opQ("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
      }
   
   } // ifMergeCR
   // finally copy back to y
   Hwf.to_array(y);
}

} // ctns

#endif
