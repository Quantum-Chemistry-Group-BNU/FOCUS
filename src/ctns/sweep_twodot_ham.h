#ifndef SWEEP_TWODOT_HAM_H
#define SWEEP_TWODOT_HAM_H

#include "oper_dict.h" 
#include "oper_combine.h"

#include <functional> // for std::function
#ifdef _OPENMP
#include <omp.h>
#endif

namespace ctns{
   
const bool debug_twodot_ham = false;
extern const bool debug_twodot_ham;

// local
template <typename Tm>
void twodot_Hdiag_local(oper_dict<Tm>& c1qops,
			oper_dict<Tm>& c2qops,
		        oper_dict<Tm>& lqops,
		        oper_dict<Tm>& rqops,
		        const double ecore,
		        qtensor4<Tm>& wf){
   const auto& Hc1 = c1qops('H')[0];
   const auto& Hc2 = c2qops('H')[0];
   const auto& Hl = lqops('H')[0];
   const auto& Hr = rqops('H')[0];
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       // local contributions: all four indices in c1/c2/l/r
	       const auto& c1blk = Hc1(bm,bm); // central->mid 
	       const auto& c2blk = Hc2(bv,bv); // 
	       const auto& lblk = Hl(br,br); // left->row 
	       const auto& rblk = Hr(bc,bc); // row->col
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) = ecore + lblk(ir,ir) + c1blk(im,im) + c2blk(iv,iv) + rblk(ic,ic);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Ol*Oc1
template <typename Tm>
void twodot_Hdiag_OlOc1(const qtensor2<Tm>& Ol,
		        const qtensor2<Tm>& Oc1,
		        qtensor4<Tm>& wf,
		        const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& lblk  = Ol(br,br); 
	       const auto& c1blk = Oc1(bm,bm); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*lblk(ir,ir)*c1blk(im,im);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Ol*Oc2
template <typename Tm>
void twodot_Hdiag_OlOc2(const qtensor2<Tm>& Ol,
		        const qtensor2<Tm>& Oc2,
		        qtensor4<Tm>& wf,
		        const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& lblk  = Ol(br,br); 
	       const auto& c2blk = Oc2(bv,bv); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*lblk(ir,ir)*c2blk(iv,iv);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Ol*Or
template <typename Tm>
void twodot_Hdiag_OlOr(const qtensor2<Tm>& Ol,
		       const qtensor2<Tm>& Or,
		       qtensor4<Tm>& wf,
		       const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& lblk = Ol(br,br); 
	       const auto& rblk = Or(bc,bc); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*lblk(ir,ir)*rblk(ic,ic);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Oc1*Oc2
template <typename Tm>
void twodot_Hdiag_Oc1Oc2(const qtensor2<Tm>& Oc1,
		         const qtensor2<Tm>& Oc2,
		         qtensor4<Tm>& wf,
		         const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& c1blk = Oc1(bm,bm); 
	       const auto& c2blk = Oc2(bv,bv); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*c1blk(im,im)*c2blk(iv,iv);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Oc1*Or
template <typename Tm>
void twodot_Hdiag_Oc1Or(const qtensor2<Tm>& Oc1,
		        const qtensor2<Tm>& Or,
		        qtensor4<Tm>& wf,
		        const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& c1blk = Oc1(bm,bm); 
	       const auto& rblk  = Or(bc,bc); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*c1blk(im,im)*rblk(ic,ic);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

// Oc2*Or
template <typename Tm>
void twodot_Hdiag_Oc2Or(const qtensor2<Tm>& Oc2,
		        const qtensor2<Tm>& Or,
		        qtensor4<Tm>& wf,
		        const Tm wt=1.0){
   for(int bm=0; bm<wf.mids(); bm++){
      for(int bv=0; bv<wf.vers(); bv++){
         for(int br=0; br<wf.rows(); br++){
            for(int bc=0; bc<wf.cols(); bc++){
               auto& blk = wf(bm,bv,br,bc);
	       if(blk.size() == 0) continue;
	       int mdim = wf.qmid.get_dim(bm);
	       int vdim = wf.qver.get_dim(bv);
	       int rdim = wf.qrow.get_dim(br);  
	       int cdim = wf.qcol.get_dim(bc);
	       const auto& c2blk = Oc2(bv,bv); 
	       const auto& rblk  = Or(bc,bc); 
	       for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           blk[iv*mdim+im](ir,ic) += wt*c2blk(iv,iv)*rblk(ic,ic);
                        } // ir
                     } // ic
                  } // im
	       } // iv
	    } // bc
	 } // br
      } // bv
   } // bm
}

template <typename Tm>
void twodot_Hdiag_BQ(const std::string& superblock,
		     const bool& ifkr,
		     oper_dict<Tm>& qops1,
		     oper_dict<Tm>& qops2,
		     qtensor4<Tm>& wf,
       	             const int size,
       	             const int rank){
   const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
   char BQ1 = ifNC? 'B' : 'Q';
   char BQ2 = ifNC? 'Q' : 'B';
   const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
   auto bindex = oper_index_opB(cindex, ifkr);
   if(rank == 0 && debug_twodot_ham){ 
      std::cout << " superblock=" << superblock << " ifNC=" << ifNC 
	        << " " << BQ1 << BQ2 << " size=" << bindex.size() << std::endl;
   }
   // B^L*Q^R or Q^L*B^R 
   for(const auto& index : bindex){
      int iproc = distribute2(index,size);
      if(iproc == rank){
         const Tm wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 due to B^H*Q^H
         const auto& O1 = qops1(BQ1).at(index);
         const auto& O2 = qops2(BQ2).at(index);
         if(O1.sym != qsym()) continue; // screening for <l|B/Q^l_{pq}|l>
	 if(superblock == "lc1"){ 
            twodot_Hdiag_OlOc1(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_OlOc1(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "lc2"){ 
            twodot_Hdiag_OlOc2(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_OlOc2(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "lr"){
            twodot_Hdiag_OlOr(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_OlOr(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "c1c2"){
            twodot_Hdiag_Oc1Oc2(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_Oc1Oc2(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "c1r"){
            twodot_Hdiag_Oc1Or(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_Oc1Or(O1.K(0),O2.K(0),wf,wt);
	 }else if(superblock == "c2r"){
            twodot_Hdiag_Oc2Or(O1,O2,wf,wt);
	    if(ifkr) twodot_Hdiag_Oc2Or(O1.K(0),O2.K(0),wf,wt);
	 }
      } // iproc
   } // index
}

template <typename Tm>
std::vector<double> twodot_Hdiag(const bool ifkr,
				 oper_dict<Tm>& c1qops,
				 oper_dict<Tm>& c2qops,
			         oper_dict<Tm>& lqops,
			         oper_dict<Tm>& rqops,
			         const double ecore,
			         qtensor4<Tm>& wf,
				 const int size,
				 const int rank){
   if(rank == 0 && debug_twodot_ham){ 
      std::cout << "ctns::twodot_Hdiag ifkr=" << ifkr 
	        << " size=" << size << std::endl;
   }
   //
   // 1. local terms: <lc1c2r|H|lc1c2r> = <lc1c2r|Hl*Ic1*Ic2*Ir+...|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
   // 
   twodot_Hdiag_local(c1qops, c2qops, lqops, rqops, ecore/size, wf);
   //
   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //
   //        B/Q^C1 B/Q^C2
   //         |      |
   // B/Q^L---*------*---B/Q^R
   //
   twodot_Hdiag_BQ("lc1" ,ifkr, lqops,c1qops,wf,size,rank);
   twodot_Hdiag_BQ("lc2" ,ifkr, lqops,c2qops,wf,size,rank);
   twodot_Hdiag_BQ("lr"  ,ifkr, lqops, rqops,wf,size,rank);
   twodot_Hdiag_BQ("c1c2",ifkr,c1qops,c2qops,wf,size,rank);
   twodot_Hdiag_BQ("c1r" ,ifkr,c1qops, rqops,wf,size,rank);
   twodot_Hdiag_BQ("c2r" ,ifkr,c2qops, rqops,wf,size,rank);
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

//
// construct H*wf: if ifkr=True, construct skeleton sigma vector 
//
//  Local terms:
//   1. H^LC1
//   2. H^C2R
//  One-index operators
//   3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c. 
//   4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c. 
//  Two-index operators
//   5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
//   6. Bps^1*Qps^2 / Qqr^1*Bqr^2
//
/* 
template <typename Tm> 
void twodot_Hx(Tm* y,
	       const Tm* x,
	       const int& isym,
	       const bool& ifkr,
	       oper_dict<Tm>& c1qops,
	       oper_dict<Tm>& c2qops,
	       oper_dict<Tm>& lqops,
	       oper_dict<Tm>& rqops,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore,
	       qtensor4<Tm>& wf,
	       const int size,
	       const int rank){
   if(rank == 0 && debug_twodot_ham){ 
      std::cout << "ctns::twodot_Hx ifkr=" << ifkr 
	        << " size=" << size << std::endl;
   }
   const bool dagger = true;
   const Tm scale = ifkr? 0.5 : 1.0;
   //
   // constant term
   //
   wf.from_array(x);
   const Tm fac = scale*(ecore/size);
   qtensor4<Tm> Hwf = fac*wf;
   //
   // Local terms:
   //
   //  1. H^LC1
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   auto Hwf1 = scale*oper_compxwf_opH("lc",wf1,lqops,c1qops,isym,ifkr,int2e,int1e,size,rank); 
   //  2. H^C2R
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   auto Hwf2 = scale*oper_compxwf_opH("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,size,rank);
   //----------------------------------------------------------------------------
   // Generic formula: L=lc1, R=c2r
   // O^lc1*O^c2r|lc1c2r>psi[lc1c2r] = O^lc1|lc1>O^c2r|c2r>(-1)^{p(lc1)*p(O^c2r)}psi[lc1c2r]
   //		      		     = O^lc1|lc1>( (-1)^{p(lc1)*p(O^c2r)} (O^c2r|c2r>psi[lc1c2r]) )
   //----------------------------------------------------------------------------
   //
   // One-index terms:
   //
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(lqops.cindex, c1qops.cindex);
   for(const auto& pr : infoC1){
      int iformula = pr.first;
      int index = pr.second;
      qtensor3<Tm> qt3n, qt3h;
      // p1^L1C1+*Sp1^C2R
      qt3n = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,size,rank);
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
      Hwf1 += oper_normxwf_opC("lc",qt3n,lqops,c1qops,iformula,index); 
      // -Sp1^C2R+*p1^L1C1
      qt3h = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,size,rank,dagger); 
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
      Hwf1 -= oper_normxwf_opC("lc",qt3h,lqops,c1qops,iformula,index,dagger);
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(c2qops.cindex, rqops.cindex);
   for(const auto& pr : infoC2){
      int iformula = pr.first;
      int index = pr.second;
      qtensor3<Tm> qt3n, qt3h;
      // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+
      qt3n = oper_normxwf_opC("cr",wf2,c2qops,rqops,iformula,index);
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
      Hwf1 -= oper_compxwf_opS("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index,size,rank);
      // Sq2^LC1+*q2^C2R
      qt3h = oper_normxwf_opC("cr",wf2,c2qops,rqops,iformula,index,dagger);
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
      Hwf1 += oper_compxwf_opS("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,size,rank,dagger);
   }
   //
   // Two-index terms:
   //  
   int slc1 = lqops.cindex.size() + c1qops.cindex.size();
   int sc2r = c2qops.cindex.size() + rqops.cindex.size();
   const bool ifNC = (slc1 <= sc2r);
   if(ifNC){
      // 5. Apq^LC1*Ppq^C2R + h.c.
      auto ainfo = oper_combine_opA(lqops.cindex, c1qops.cindex, ifkr);
      for(const auto pr : ainfo){
	 int iformula = pr.first;
	 int index = pr.second;
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacAP(index) : 1.0;
	    qtensor3<Tm> qt3n, qt3h;
   	    // Apq*Ppq
	    qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
            qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
            Hwf1 += wt*oper_normxwf_opA("lc",qt3n,lqops,c1qops,ifkr,iformula,index);
	    // (Apq*Ppq)^H
	    qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
            qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
            Hwf1 += wt*oper_normxwf_opA("lc",qt3h,lqops,c1qops,ifkr,iformula,index,dagger);
         } // iproc
      }
      // 6. Bps^LC1*Qps^C2R + h.c.
      auto binfo = oper_combine_opB(lqops.cindex, c1qops.cindex, ifkr);
      for(const auto pr : binfo){
	 int iformula = pr.first;
	 int index = pr.second;
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	    qtensor3<Tm> qt3n, qt3h;
   	    // Bpq*Qpq
	    qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
            qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
            Hwf1 += wt*oper_normxwf_opB("lc",qt3n,lqops,c1qops,ifkr,iformula,index);
	    // (Bpq*Qpq)^H
	    qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
            qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
            Hwf1 += wt*oper_normxwf_opB("lc",qt3h,lqops,c1qops,ifkr,iformula,index,dagger);
         } // iproc
      }
   }else{
      // 5. Ars^C2R*Prs^LC1 + h.c.
      auto ainfo = oper_combine_opA(c2qops.cindex, rqops.cindex, ifkr);
      for(const auto pr : ainfo){
	 int iformula = pr.first;
	 int index = pr.second;
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacAP(index) : 1.0;
	    qtensor3<Tm> qt3n, qt3h;
	    // Prs*Ars
            qt3n = oper_normxwf_opA("cr",wf2,c2qops,rqops,ifkr,iformula,index);
            qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
	    Hwf1 += wt*oper_compxwf_opP("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	    // (Prs*Ars)^H
            qt3h = oper_normxwf_opA("cr",wf2,c2qops,rqops,ifkr,iformula,index,dagger);
            qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
	    Hwf1 += wt*oper_compxwf_opP("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
	 } // iproc
      }
      // 6. Qqr^LC1*Bqr^C2R
      auto binfo = oper_combine_opB(c2qops.cindex, rqops.cindex, ifkr);
      for(const auto pr : binfo){
	 int iformula = pr.first;
	 int index = pr.second;
         int iproc = distribute2(index,size);
	 if(iproc == rank){
	    const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	    qtensor3<Tm> qt3n, qt3h;
	    // Prs*Ars
            qt3n = oper_normxwf_opB("cr",wf2,c2qops,rqops,ifkr,iformula,index);
            qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
	    Hwf1 += wt*oper_compxwf_opQ("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	    // (Prs*Ars)^H
            qt3h = oper_normxwf_opB("cr",wf2,c2qops,rqops,ifkr,iformula,index,dagger);
            qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
	    Hwf1 += wt*oper_compxwf_opQ("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
	 } // iproc
      }
   } // ifNC
   Hwf += Hwf1.split_c2r(wf.qver,wf.qcol);
   Hwf += Hwf2.split_lc1(wf.qrow,wf.qmid); 
   Hwf.to_array(y);
}
*/


template <typename Tm>
qtensor3<Tm> twodot_Hx_CS(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   qtensor3<Tm> qt3n, qt3h;
   // p1^L1C1+*Sp1^C2R
   qt3n = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,size,rank);
   qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
   auto Hwf1 = oper_normxwf_opC("lc",qt3n,lqops,c1qops,iformula,index); 
   // -Sp1^C2R+*p1^L1C1
   qt3h = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,size,rank,dagger); 
   qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
   Hwf1 -= oper_normxwf_opC("lc",qt3h,lqops,c1qops,iformula,index,dagger);
   return Hwf1;
}
 
template <typename Tm>
qtensor3<Tm> twodot_Hx_SC(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   qtensor3<Tm> qt3n, qt3h;
   // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+
   qt3n = oper_normxwf_opC("cr",wf2,c2qops,rqops,iformula,index);
   qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
   auto Hwf1 = -oper_compxwf_opS("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index,size,rank);
   // Sq2^LC1+*q2^C2R
   qt3h = oper_normxwf_opC("cr",wf2,c2qops,rqops,iformula,index,dagger);
   qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid);
   Hwf1 += oper_compxwf_opS("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,size,rank,dagger);
   return Hwf1;
}

template <typename Tm>
qtensor3<Tm> twodot_Hx_AP(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   const Tm wt = ifkr? wfacAP(index) : 1.0;
   qtensor3<Tm> qt3n, qt3h;
   // Apq*Ppq
   qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
   qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
   auto Hwf1 = wt*oper_normxwf_opA("lc",qt3n,lqops,c1qops,ifkr,iformula,index);
   // (Apq*Ppq)^H
   qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
   Hwf1 += wt*oper_normxwf_opA("lc",qt3h,lqops,c1qops,ifkr,iformula,index,dagger);
   return Hwf1;
}

template <typename Tm>
qtensor3<Tm> twodot_Hx_BQ(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   const Tm wt = ifkr? wfacBQ(index) : wfac(index);
   qtensor3<Tm> qt3n, qt3h;
   // Bpq*Qpq
   qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
   qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
   auto Hwf1 = wt*oper_normxwf_opB("lc",qt3n,lqops,c1qops,ifkr,iformula,index);
   // (Bpq*Qpq)^H
   qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
   Hwf1 += wt*oper_normxwf_opB("lc",qt3h,lqops,c1qops,ifkr,iformula,index,dagger);
   return Hwf1;
}

template <typename Tm>
qtensor3<Tm> twodot_Hx_PA(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   const Tm wt = ifkr? wfacAP(index) : 1.0;
   qtensor3<Tm> qt3n, qt3h;
   // Prs*Ars
   qt3n = oper_normxwf_opA("cr",wf2,c2qops,rqops,ifkr,iformula,index);
   qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
   auto Hwf1 = wt*oper_compxwf_opP("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
   // (Prs*Ars)^H
   qt3h = oper_normxwf_opA("cr",wf2,c2qops,rqops,ifkr,iformula,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
   Hwf1 += wt*oper_compxwf_opP("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
   return Hwf1;
}

template <typename Tm>
qtensor3<Tm> twodot_Hx_QB(const int iformula,
		  	  const int index,
	          	  const int& isym,
	          	  const bool& ifkr,
	          	  oper_dict<Tm>& c1qops,
	          	  oper_dict<Tm>& c2qops,
	          	  oper_dict<Tm>& lqops,
	          	  oper_dict<Tm>& rqops,
	          	  const integral::two_body<Tm>& int2e,
	          	  const integral::one_body<Tm>& int1e,
	          	  const qtensor4<Tm>& wf,
	          	  const int& size,
	          	  const int& rank){
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   const bool dagger = true;
   const Tm wt = ifkr? wfacBQ(index) : wfac(index);
   qtensor3<Tm> qt3n, qt3h;
   // Prs*Ars
   qt3n = oper_normxwf_opB("cr",wf2,c2qops,rqops,ifkr,iformula,index);
   qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid); 
   auto Hwf1 = wt*oper_compxwf_opQ("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
   // (Prs*Ars)^H
   qt3h = oper_normxwf_opB("cr",wf2,c2qops,rqops,ifkr,iformula,index,dagger);
   qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid); 
   Hwf1 += wt*oper_compxwf_opQ("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
   return Hwf1;
}

template <typename Tm>
qtensor3<Tm> twodot_Hx_local(const int& isym,
	          	     const bool& ifkr,
	          	     oper_dict<Tm>& c1qops,
	          	     oper_dict<Tm>& c2qops,
	          	     oper_dict<Tm>& lqops,
	          	     oper_dict<Tm>& rqops,
	          	     const integral::two_body<Tm>& int2e,
	          	     const integral::one_body<Tm>& int1e,
			     const qtensor4<Tm>& wf,
	          	     const int& size,
	          	     const int& rank){
   const Tm scale = ifkr? 0.5 : 1.0;
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   // H^LC1
   auto Hwf1 = scale*oper_compxwf_opH("lc",wf1,lqops,c1qops,isym,ifkr,int2e,int1e,size,rank); 
   // H^C2R
   auto Hwf2 = scale*oper_compxwf_opH("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,size,rank);
   Hwf1 += Hwf2.split_lc1(wf.qrow,wf.qmid).merge_c2r(); 
   return Hwf1; 
} 

// Collect all Hx_funs 
template <typename Tm>
Hx_functors<Tm> twodot_Hx_functors(const int& isym,
	                           const bool& ifkr,
	                           oper_dict<Tm>& c1qops,
	                           oper_dict<Tm>& c2qops,
	                           oper_dict<Tm>& lqops,
	                           oper_dict<Tm>& rqops,
	                           const integral::two_body<Tm>& int2e,
	                           const integral::one_body<Tm>& int1e,
	                           const qtensor4<Tm>& wf,
	                           const int& size,
	                           const int& rank){
   Hx_functors<Tm> Hx_funs;
   // Local terms:
   Hx_functor<Tm> Hx("Hloc", 0, 0);
   Hx.opxwf = bind(&twodot_Hx_local<Tm>, std::cref(isym), std::cref(ifkr),
		   std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops),
		   std::cref(int2e), std::cref(int1e), std::cref(wf),
		   std::cref(size), std::cref(rank));
   Hx_funs.push_back(Hx); 
   // One-index terms:
   // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   auto infoC1 = oper_combine_opC(lqops.cindex, c1qops.cindex);
   for(const auto& pr : infoC1){
      int iformula = pr.first;
      int index = pr.second;
      Hx_functor<Tm> Hx("CS", iformula, index);
      Hx.opxwf = bind(&twodot_Hx_CS<Tm>, iformula, index, std::cref(isym), std::cref(ifkr),
           	      std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops), 
           	      std::cref(int2e), std::cref(int1e), std::cref(wf),
                      std::cref(size), std::cref(rank));
      Hx_funs.push_back(Hx); 
   }
   // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   auto infoC2 = oper_combine_opC(c2qops.cindex, rqops.cindex);
   for(const auto& pr : infoC2){
      int iformula = pr.first;
      int index = pr.second;
      Hx_functor<Tm> Hx("SC", iformula, index);
      Hx.opxwf = bind(&twodot_Hx_SC<Tm>, iformula, index, std::cref(isym), std::cref(ifkr),
           	      std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops), 
           	      std::cref(int2e), std::cref(int1e), std::cref(wf), 
		      std::cref(size), std::cref(rank));
      Hx_funs.push_back(Hx); 
   }
   // Two-index terms:
   int slc1 = lqops.cindex.size() + c1qops.cindex.size();
   int sc2r = c2qops.cindex.size() + rqops.cindex.size();
   const bool ifNC = (slc1 <= sc2r);
   auto ainfo = ifNC? oper_combine_opA(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opA(c2qops.cindex, rqops.cindex, ifkr);
   auto binfo = ifNC? oper_combine_opB(lqops.cindex, c1qops.cindex, ifkr) :
      		      oper_combine_opB(c2qops.cindex, rqops.cindex, ifkr);
   auto afun = ifNC? &twodot_Hx_AP<Tm> : &twodot_Hx_PA<Tm>; 
   auto bfun = ifNC? &twodot_Hx_BQ<Tm> : &twodot_Hx_QB<Tm>;
   auto alabel = ifNC? "AP" : "PA";
   auto blabel = ifNC? "BQ" : "QB"; 
   // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
   for(const auto pr : ainfo){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
         Hx_functor<Tm> Hx(alabel, iformula, index);
         Hx.opxwf = bind(afun, iformula, index, std::cref(isym), std::cref(ifkr),
              	    std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops), 
              	    std::cref(int2e), std::cref(int1e), std::cref(wf),
		    std::cref(size), std::cref(rank));
         Hx_funs.push_back(Hx); 
      } // iproc
   }
   // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
   for(const auto pr : binfo){
      int iformula = pr.first;
      int index = pr.second;
      int iproc = distribute2(index,size);
      if(iproc == rank){
         Hx_functor<Tm> Hx(blabel, iformula, index);
         Hx.opxwf = bind(bfun, iformula, index, std::cref(isym), std::cref(ifkr),
              	    std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops), 
              	    std::cref(int2e), std::cref(int1e), std::cref(wf),
		    std::cref(size), std::cref(rank));
         Hx_funs.push_back(Hx);
      } // iproc
   }
   // debug
   if(rank == 0){
      std::cout << "twodot_Hx_functors: size=" << Hx_funs.size() 
                << " CS:" << infoC1.size()
                << " SC:" << infoC2.size()
                << " " << alabel << ":" << ainfo.size()
                << " " << blabel << ":" << binfo.size()
                << std::endl; 
      const bool debug = false;
      if(debug){
         for(int i=0; i<Hx_funs.size(); i++){
            std::cout << "i=" << i << Hx_funs[i] << std::endl;
         } // i
      }
   }
   return Hx_funs;
}

template <typename Tm> 
void twodot_Hx(Tm* y,
	       const Tm* x,
	       qtensor4<Tm>& wf,
	       Hx_functors<Tm>& Hx_funs,
 	       const bool ifkr,
	       const double ecore,
	       const int size,
	       const int rank){
   const int maxthreads = omp_get_max_threads();
   if(rank == 0 && debug_twodot_ham){ 
      std::cout << "ctns::twodot_Hx size=" << size 
                << " maxthreads=" << maxthreads
                << std::endl;
   }
   wf.from_array(x);
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   const Tm scale = ifkr? 0.5 : 1.0;
   const Tm fac = scale*(ecore/size);
   qtensor4<Tm> Hwf = fac*wf;
   //=======================
   // Parallel evaluation
   //=======================
   // initialization
   std::vector<qtensor3<Tm>> Hwf1_lst(maxthreads);
   for(int i=0; i<maxthreads; i++){
      Hwf1_lst[i].init(wf1.sym, wf1.qmid, wf1.qrow, wf1.qcol, wf1.dir);
   }
   // compute
   #pragma omp parallel for schedule(dynamic)
   for(int i=0; i<Hx_funs.size(); i++){
      int omprank = omp_get_thread_num();
      Hwf1_lst[omprank] += Hx_funs[i]();
   }
   // reduction & save
   for(int i=0; i<maxthreads; i++){
      Hwf += Hwf1_lst[i].split_c2r(wf.qver,wf.qcol);
   }
   Hwf.to_array(y);
}

} // ctns

#endif
