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
void onedot_Hdiag_BQ(const std::string& superblock,
		     const bool& ifkr,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     qtensor3<Tm>& wf,
       	             const int size,
       	             const int rank){
   const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
   auto bindex = oper_index_opB(cindex, ifkr);
   if(rank == 0 && debug_onedot_ham){ 
      std::cout << " superblock=" << superblock << " ifNC=" << ifNC 
	        << " " << BQ1 << BQ2 << " size=" << bindex.size() 
		<< std::endl;
   }
   // B^L*Q^R or Q^L*B^R 
   for(const auto& index : bindex){
      int iproc = distribute2(index,size);
      if(iproc == rank){
         const Tm wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 due to B^H*Q^H
         const auto& O1 = qops1(BQ1).at(index);
         const auto& O2 = qops2(BQ2).at(index);
         if(O1.sym != qsym()) continue; // screening for <l|B/Q^l_{pq}|l>
	 if(superblock == "lc"){ 
            onedot_Hdiag_OlOc(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OlOc(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "cr"){
            onedot_Hdiag_OcOr(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OcOr(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "lr"){
            onedot_Hdiag_OlOr(O1,O2,wf,wt);
	    if(ifkr) onedot_Hdiag_OlOr(O1.K(0),O2.K(0),wf,wt);
	 }
      }
   } // index
}
	
template <typename Tm>
std::vector<double> onedot_Hdiag(const bool& ifkr,
				 oper_dict<Tm>& cqops,
			         oper_dict<Tm>& lqops,
			         oper_dict<Tm>& rqops,
			         const double ecore,
			         qtensor3<Tm>& wf,
	       	                 const int size,
	       	                 const int rank){
   if(rank == 0 && debug_onedot_ham){ 
      std::cout << "ctns::onedot_Hdiag ifkr=" << ifkr 
	        << " size=" << size << std::endl;
   }
   //
   // 1. local terms: <lcr|H|lcr> = <lcr|Hl*Ic*Ir+...|lcr> = Hll + Hcc + Hrr
   //
   onedot_Hdiag_local(cqops, lqops, rqops, ecore/size, wf);
   //
   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //
   //         B/Q^C
   //         |
   // B/Q^L---*---B/Q^R
   //
   onedot_Hdiag_BQ("lc",ifkr,lqops,cqops,wf,size,rank);
   onedot_Hdiag_BQ("cr",ifkr,cqops,rqops,wf,size,rank);
   onedot_Hdiag_BQ("lr",ifkr,lqops,rqops,wf,size,rank);
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
	       qtensor3<Tm>& wf,
	       const int size,
	       const int rank){
   if(rank == 0 && debug_onedot_ham){ 
      std::cout << "ctns::onedot_Hx ifkr=" << ifkr 
	        << " size=" << size << std::endl;
   }
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
   const bool ifNC = lqops.cindex.size() < rqops.cindex.size();
   if(ifNC){
      /*
         Generic formula: L=l, R=cr: A[l]*P[cr]+B[l]*Q[cr]
         O^l*O^cr|lcr>psi[lcr]=O^l|l>O^cr|cr>(-1)^{p(l)*p(O^cr)}psi[lcr]
			      =O^l|l>( (-1)^{p(l)*p(O^cr)} (O^cr|cr>psi[lcr]) )
      */
      // 1. H^l 
      Hwf += scale*contract_qt3_qt2_l(wf,lqops('H')[0]);
      // 2. H^cr
      Hwf += scale*oper_compxwf_opH("cr",wf,cqops,rqops,isym,ifkr,int2e,int1e,size,rank);
      // 3. p1^l+*Sp1^cr + h.c.
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
      for(const auto& pr : infoC){
         int iformula = pr.first;
         int index = pr.second;
	 const auto& op1 = lqops('S').at(index);
         auto qt3n = oper_normxwf_opC("cr",wf,cqops,rqops,iformula,index);
	 auto qt3h = oper_normxwf_opC("cr",wf,cqops,rqops,iformula,index,dagger);
	 Hwf -= oper_kernel_OIwf("lc",qt3n.row_signed(),op1); 
	 Hwf += oper_kernel_OIwf("lc",qt3h.row_signed(),op1,dagger); 
      }
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
   }else{
      /*
         Generic formula: L=lc, R=r: A[lc]*P[r]+B[lc]*Q[r]
         O^lc*O^r|lcr>psi[lcr]=O^lc|lc>O^r|r>(-1)^{p(lc)*p(O^r)}psi[lcr]
			      =O^lc|lc>( (-1)^{p(l)*p(O^cr)} ((-1)^{p(c)*p(O^cr)} O^r|c>psi[lcr]) )
      */
      // 1. H^lc
      Hwf += scale*oper_compxwf_opH("lc",wf,lqops,cqops,isym,ifkr,int2e,int1e,size,rank);
      // 2. H^r
      Hwf += scale*contract_qt3_qt2_r(wf,rqops('H')[0]);
      // 3. p1^lc+*Sp1^r + h.c.
      auto infoC = oper_combine_opC(lqops.cindex, cqops.cindex);
      for(const auto& pr : infoC){
         int iformula = pr.first;
         int index = pr.second;
	 const auto& op2 = rqops('S').at(index);
         auto qt3n = oper_kernel_IOwf("cr",wf,op2,1); // p(c) is taken into account in IOwf
	 auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
         Hwf += oper_normxwf_opC("lc",qt3n.row_signed(),lqops,cqops,iformula,index); // p(l) is taken into account
	 Hwf -= oper_normxwf_opC("lc",qt3h.row_signed(),lqops,cqops,iformula,index,dagger);
      }
      // 4. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
      for(const auto& op2C : rqops('C')){
         int q2 = op2C.first;
  	 const auto& op2 = op2C.second;
	 auto qt3n = oper_kernel_IOwf("cr",wf,op2,1); 
	 auto qt3h = oper_kernel_IOwf("cr",wf,op2,1,dagger);
	 Hwf -= oper_compxwf_opS("lc",qt3n.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2,size,rank);
	 Hwf += oper_compxwf_opS("lc",qt3h.row_signed(),lqops,cqops,isym,ifkr,int2e,int1e,q2,size,rank,dagger);
      }
      // 5. Ars^r*Prs^lc + h.c.
      auto aindex = oper_index_opA(rqops.cindex, ifkr);
      for(const auto& index : aindex){
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacAP(index) : 1.0;
	    const auto& op2 = rqops('A').at(index); 
	    auto qt3n = wt*oper_kernel_IOwf("cr",wf,op2,0);
	    auto qt3h = wt*oper_kernel_IOwf("cr",wf,op2,0,dagger);
	    Hwf += oper_compxwf_opP("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	    Hwf += oper_compxwf_opP("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
	 }
      }
      // 6. Qqr^lc*Bqr^r (using Hermicity)
      auto bindex = oper_index_opB(rqops.cindex, ifkr);
      for(const auto& index : bindex){
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	    const auto& op2 = rqops('B').at(index);
	    auto qt3n = wt*oper_kernel_IOwf("cr",wf,op2,0);
	    auto qt3h = wt*oper_kernel_IOwf("cr",wf,op2,0,dagger);
	    Hwf += oper_compxwf_opQ("lc",qt3n,lqops,cqops,isym,ifkr,int2e,int1e,index);
	    Hwf += oper_compxwf_opQ("lc",qt3h,lqops,cqops,isym,ifkr,int2e,int1e,index,dagger);
	 }
      }
   } // ifNC
   Hwf.to_array(y);
}

} // ctns

#endif
