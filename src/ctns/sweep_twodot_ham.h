#ifndef SWEEP_TWODOT_HAM_H
#define SWEEP_TWODOT_HAM_H

namespace ctns{
   
const bool debug_twodot_ham = true;
extern const bool debug_twodot_ham;

// local 
template <typename Tm>
void twodot_Hdiag_local(oper_dict<Tm>& c1qops,
			oper_dict<Tm>& c2qops,
		        oper_dict<Tm>& lqops,
		        oper_dict<Tm>& rqops,
		        const double ecore,
		        qtensor4<Tm>& wf){
   const auto& Hc1 = c1qops['H'][0];
   const auto& Hc2 = c2qops['H'][0];
   const auto& Hl = lqops['H'][0];
   const auto& Hr = rqops['H'][0];
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
	       // 1. local contributions: all four indices in c1/c2/l/r
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
std::vector<double> twodot_Hdiag(const bool ifkr,
				 oper_dict<Tm>& c1qops,
				 oper_dict<Tm>& c2qops,
			         oper_dict<Tm>& lqops,
			         oper_dict<Tm>& rqops,
			         const double ecore,
			         qtensor4<Tm>& wf){
   if(debug_twodot_ham) std::cout << "ctns::twodot_Hdiag ifkr=" << ifkr << std::endl;
   //
   // 1. local terms: <lc1c2r|H|lc1c2r> = <lc1c2r|Hl*Ic1*Ic2*Ir+...|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
   // 
   twodot_Hdiag_local(c1qops, c2qops, lqops, rqops, ecore, wf);
   //
   // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
   //
   //         B^C1 Q/B^C2
   //         |     |
   // B/Q^L---*-----*---Q/B^R
   //
   if(not ifkr){
      // Q^L*B^C1 
      for(const auto& p : c1qops['B']){
         const auto& Bc1 = p.second;
         if(Bc1.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql = lqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_OlOc1(Ql,Bc1,wf,wt);
      } 
      // Q^L*B^C2 
      for(const auto& p : c2qops['B']){
         const auto& Bc2 = p.second;
         if(Bc2.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql = lqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_OlOc2(Ql,Bc2,wf,wt);
      } 
      // B^C1*Q^R 
      for(const auto& p : c1qops['B']){
         const auto& Bc1 = p.second;
         if(Bc1.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr = rqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc1Or(Bc1,Qr,wf,wt);
      }
      // B^C1*Q^C2 
      for(const auto& p : c1qops['B']){
         const auto& Bc1 = p.second;
         if(Bc1.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qc2 = c2qops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc1Oc2(Bc1,Qc2,wf,wt);
      }
      // B^C2*Q^R 
      for(const auto& p : c2qops['B']){
         const auto& Bc2 = p.second;
         if(Bc2.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr = rqops['Q'].at(p.first);
	 const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc2Or(Bc2,Qr,wf,wt);
      }
      // B^L*Q^R or Q^L*B^R 
      if(lqops['B'].size() <= rqops['B'].size()){
         for(const auto& p : lqops['B']){
            const auto& Bl = p.second;
            if(Bl.sym != qsym()) continue;
            const auto& Qr = rqops['Q'].at(p.first);
	    const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
            twodot_Hdiag_OlOr(Bl,Qr,wf,wt);
         } 
      }else{
         for(const auto& p: rqops['B']){
            const auto& Br = p.second;
            if(Br.sym != qsym()) continue;
            const auto& Ql = lqops['Q'].at(p.first);
	    const Tm wt = 2.0*wfac(p.first); // taking into account B^d*Q^d
            twodot_Hdiag_OlOr(Ql,Br,wf,wt);
         }
      }
   }else{
      // Q^L*B^C1 
      for(const auto& p : c1qops['B']){
         const auto& Bc1_A = p.second;
         if(Bc1_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql_A = lqops['Q'].at(p.first);
	 const auto& Bc1_B = Bc1_A.K(0);
	 const auto& Ql_B  = Ql_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_OlOc1(Ql_A,Bc1_A,wf,wt);
         twodot_Hdiag_OlOc1(Ql_B,Bc1_B,wf,wt);
      }
      // Q^L*B^C2 
      for(const auto& p : c2qops['B']){
         const auto& Bc2_A = p.second;
         if(Bc2_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Ql_A = lqops['Q'].at(p.first);
	 const auto& Bc2_B = Bc2_A.K(0);
	 const auto& Ql_B  = Ql_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_OlOc2(Ql_A,Bc2_A,wf,wt);
         twodot_Hdiag_OlOc2(Ql_B,Bc2_B,wf,wt);
      }
      // B^C1*Q^R 
      for(const auto& p : c1qops['B']){
         const auto& Bc1_A = p.second;
         if(Bc1_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr_A = rqops['Q'].at(p.first);
	 const auto& Bc1_B = Bc1_A.K(0);
	 const auto& Qr_B  = Qr_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc1Or(Bc1_A,Qr_A,wf,wt);
         twodot_Hdiag_Oc1Or(Bc1_B,Qr_B,wf,wt);
      }
      // B^C1*Q^C2 
      for(const auto& p : c1qops['B']){
         const auto& Bc1_A = p.second;
         if(Bc1_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qc2_A = c2qops['Q'].at(p.first);
	 const auto& Bc1_B = Bc1_A.K(0);
	 const auto& Qc2_B = Qc2_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc1Oc2(Bc1_A,Qc2_A,wf,wt);
         twodot_Hdiag_Oc1Oc2(Bc1_B,Qc2_B,wf,wt);
      }
      // B^C2*Q^R 
      for(const auto& p : c2qops['B']){
         const auto& Bc2_A = p.second;
         if(Bc2_A.sym != qsym()) continue; // screening for <l|B^C_{pq}|l>
         const auto& Qr_A = rqops['Q'].at(p.first);
	 const auto& Bc2_B = Bc2_A.K(0);
	 const auto& Qr_B  = Qr_A.K(0);
	 const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
         twodot_Hdiag_Oc2Or(Bc2_A,Qr_A,wf,wt);
         twodot_Hdiag_Oc2Or(Bc2_B,Qr_B,wf,wt);
      }
      // B^L*Q^R or Q^L*B^R 
      if(lqops['B'].size() <= rqops['B'].size()){
         for(const auto& p : lqops['B']){
            const auto& Bl_A = p.second;
            if(Bl_A.sym != qsym()) continue;
            const auto& Qr_A = rqops['Q'].at(p.first);
	    const auto& Bl_B = Bl_A.K(0);
	    const auto& Qr_B = Qr_A.K(0);
	    const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
            twodot_Hdiag_OlOr(Bl_A,Qr_A,wf,wt);
            twodot_Hdiag_OlOr(Bl_B,Qr_B,wf,wt);
         } 
      }else{
         for(const auto& p: rqops['B']){
            const auto& Br_A = p.second;
            if(Br_A.sym != qsym()) continue;
            const auto& Ql_A = lqops['Q'].at(p.first);
	    const auto& Br_B = Br_A.K(0);
	    const auto& Ql_B = Ql_A.K(0);
	    const Tm wt = 2.0*wfacBQ(p.first); // taking into account B^d*Q^d
            twodot_Hdiag_OlOr(Ql_A,Br_A,wf,wt);
            twodot_Hdiag_OlOr(Ql_B,Br_B,wf,wt);
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
void twodot_Hx(Tm* y,
	       const Tm* x,
	       const int& isym,
	       const bool& ifkr,
	       const bool& ifln,
	       oper_dict<Tm>& c1qops,
	       oper_dict<Tm>& c2qops,
	       oper_dict<Tm>& lqops,
	       oper_dict<Tm>& rqops,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore,
	       qtensor4<Tm>& wf){
   if(debug_onedot_ham) std::cout << "ctns::twodot_Hx ifkr=" << ifkr << std::endl;
   const bool dagger = true;
   //
   // constant term
   //
   const Tm scale = ifkr? 0.5 : 1.0;
   wf.from_array(x);
   qtensor4<Tm> Hwf = (scale*ecore)*wf;
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
   // Local terms:
   //  1. H^LC1
   auto wf1 = wf.merge_c2r(); // wf1[l,c1,c2r]
   auto Hwf1 = scale*oper_compxwf_opH("lc",wf1,lqops,c1qops,isym,ifkr,int2e,int1e); 
   //  2. H^C2R
   auto wf2 = wf.merge_lc1(); // wf2[lc1,c2,r]
   auto Hwf2 = scale*oper_compxwf_opH("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e);
   // 
   // One-index terms:
   //  3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
   for(const auto& op1C : lqops['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      // p1^L1+*Sp1^C2R
      auto qt3n = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,p1);
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_OIwf("lc",qt3n,op1); // both lc/lr can work 
      // -Sp1^C2R+*p1^L1
      auto qt3h = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,p1,dagger); 
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_OIwf("lc",qt3h,op1,dagger);
   }
   for(const auto& op1C : c1qops['C']){
      int p1 = op1C.first;
      const auto& op1 = op1C.second;
      // p1^C1+*Sp1^C2R
      auto qt3n = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,p1);
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_kernel_IOwf("lc",qt3n,op1,1); 
      // -Sp1^C2R+*p1^C1
      auto qt3h = oper_compxwf_opS("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,p1,dagger); 
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_kernel_IOwf("lc",qt3h,op1,1,dagger);
   }
   //  4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
   for(const auto& op2C : c2qops['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      // q2^C2+*Sq2^LC1 = -Sq2^LC1*q2^C2+
      auto qt3n = oper_kernel_OIwf("cr",wf2,op2); // tmp[lc1,c2,r] 
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second); // tmp[lc1,c2r]->tmp[l,c1,c2r]
      Hwf1 -= oper_compxwf_opS("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,q2);
      // Sq2^LC1+*q2^C2
      auto qt3h = oper_kernel_OIwf("cr",wf2,op2,dagger);
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_compxwf_opS("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,q2,dagger);
   }
   for(const auto& op2C : rqops['C']){
      int q2 = op2C.first;
      const auto& op2 = op2C.second;
      // q2^R+*Sq2^LC1 = -Sq2^LC1*q2^R+
      auto qt3n = oper_kernel_IOwf("cr",wf2,op2,1); 
      qt3n = qt3n.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 -= oper_compxwf_opS("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,q2);
      // Sq2^LC1+*q2^R
      auto qt3h = oper_kernel_IOwf("cr",wf2,op2,1,dagger);
      qt3h = qt3h.row_signed().merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
      Hwf1 += oper_compxwf_opS("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,q2,dagger);
   }
   //
   // Two-index terms:
   //  5. Apq^1*Ppq^2 + h.c. / Prs^1+Ars^2+ + h.c.
   //  6. Bps^1*Qps^2 / Qqr^1*Bqr^2
   if(ifln){
      // 5. Apq^LC1*Ppq^C2R + h.c.
      // Apq^L*Ppq^C2R
      for(const auto& op1A : lqops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 const Tm wt = ifkr? wfacAP(index) : 1.0;
	 // Apq*Ppq
	 auto qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second); 
	 Hwf1 += wt*oper_kernel_OIwf("lc",qt3n,op1);
	 // (Apq*Ppq)^H
	 auto qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second); 
	 Hwf1 += wt*oper_kernel_OIwf("lc",qt3h,op1,dagger);
      }
      // Apq^C1*Ppq^C2R
      for(const auto& op1A : c1qops['A']){
	 int index = op1A.first;
	 const auto& op1 = op1A.second;
	 const Tm wt = ifkr? wfacAP(index) : 1.0;
	 // Apq*Ppq
	 auto qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second); 
	 Hwf1 += wt*oper_kernel_IOwf("lc",qt3n,op1,0);
	 // (Apq*Ppq)^H
	 auto qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second); 
	 Hwf1 += wt*oper_kernel_IOwf("lc",qt3h,op1,0,dagger);
      }
      // 6. Bps^LC1*Qps^C2R
      // Bps^L
      for(const auto& op1B : lqops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	 auto qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += wt*oper_kernel_OIwf("lc",qt3n,op1);
	 auto qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += wt*oper_kernel_OIwf("lc",qt3h,op1,dagger);
      }
      // Bps^C1
      for(const auto& op1B : c1qops['B']){
	 int index = op1B.first;
	 const auto& op1 = op1B.second;
	 const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	 auto qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += wt*oper_kernel_IOwf("lc",qt3n,op1,0);
	 auto qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += wt*oper_kernel_IOwf("lc",qt3h,op1,0,dagger);
      }
      // 
      // Cross terms: should be consistent with oper_renorm.h
      //
      if(not ifkr){
/*
         // Apq = p^L+*q^C1+ (p<q) or p^C1+*q^L+ (p<q)
         for(const auto& op1C : lqops['C']){
            int p1 = op1C.first;
	    const auto& op1 = op1C.second;
            for(const auto& op2C : c1qops['C']){
               int p2 = op2C.first;
	       const auto& op2 = op2C.second;
	       int index = (p1<p2)? oper_pack(p1,p2) : oper_pack(p2,p1);
	       double sgn = (p1<p2)? 1.0 : -1.0;
               // Apq^LC1*Ppq^C2R
	       auto qt3n = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
               qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
               Hwf1 += sgn*oper_kernel_OOwf("lc",qt3n,op1,op2,1); 
               auto qt3h = oper_compxwf_opP("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
               qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
               Hwf1 -= sgn*oper_kernel_OOwf("lc",qt3h,op1,op2,1,dagger); 
            }
         }
         // Bps = p^L+*s^C1 or p^C1+*s^L = -s^L*p^C1+  
         for(const auto& op1C : lqops['C']){
            int p1 = op1C.first;
            for(const auto& op2C : c1qops['C']){
               int p2 = op2C.first;
	       const auto& op1 = (p1<p2)? op1C.second : op1C.second.H();
	       const auto& op2 = (p1<p2)? op2C.second.H() : op2C.second;
               int index = (p1<p2)? oper_pack(p1,p2) : oper_pack(p2,p1);
	       double sgn = (p1<p2)? 1.0 : -1.0;
	       // Bps^LC1*Qps^C2R
	       auto qt3n = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index);
               qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	       Hwf1 += sgn*oper_kernel_OOwf("lc",qt3n,op1,op2,1);
	       // (Bps^LC1*Qps^C2R)^H
	       auto qt3h = oper_compxwf_opQ("cr",wf2,c2qops,rqops,isym,ifkr,int2e,int1e,index,dagger);
               qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	       Hwf1 += sgn*oper_kernel_OOwf("lc",qt3h,op1,op2,1,dagger);
            }
         }
*/
      } // ifkr

   }else{

      // 5. Ars^C2R*Prs^LC1 + h.c.
      // Ars^C2*Prs^LC1 = Prs^LC1*Ars^C2
      for(const auto& op2A : c2qops['A']){
	 int index = op2A.first;
	 const auto& op2 = op2A.second;
	 const Tm wt = ifkr? wfacAP(index) : 1.0;
	 // Ars^c2*Prs^lc1 = Prs^lc1*Ars^c2
	 auto qt3n = wt*oper_kernel_OIwf("cr",wf2,op2);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opP("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	 // (Prs^lc1*Ars^c2)^H
	 auto qt3h = wt*oper_kernel_OIwf("cr",wf2,op2,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opP("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
      }
      // Ars^R*Prs^LC1 = Prs^LC1*Ars^R
      for(const auto& op2A : rqops['A']){
	 int index = op2A.first;
	 const auto& op2 = op2A.second;
	 const Tm wt = ifkr? wfacAP(index) : 1.0;
	 // Ars^r*Prs^lc1 = Prs^lc1*Ars^r
	 auto qt3n = wt*oper_kernel_IOwf("cr",wf2,op2,0);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opP("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	 // (Prs^lc1*Ars^r)^H
	 auto qt3h = wt*oper_kernel_IOwf("cr",wf2,op2,0,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opP("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
      }
/*
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
*/
      // 6. Qqr^LC1*Bqr^C2R
      // Qqr^LC1*Bqr^C2
      for(const auto& op2B : c2qops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	 // Qqr^LC1*Bqr^C2 
	 auto qt3n = wt*oper_kernel_OIwf("cr",wf2,op2);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opQ("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	 // (Qqr^LC1*Bqr^C2)^H
	 auto qt3h = wt*oper_kernel_OIwf("cr",wf2,op2,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opQ("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
      }
      // Qqr^LC1*Bqr^R
      for(const auto& op2B : rqops['B']){
	 int index = op2B.first;
	 const auto& op2 = op2B.second;
	 const Tm wt = ifkr? wfacBQ(index) : wfac(index);
	 // Qqr^LC1*Bqr^R
	 auto qt3n = wt*oper_kernel_IOwf("cr",wf2,op2,0);
	 qt3n = qt3n.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opQ("lc",qt3n,lqops,c1qops,isym,ifkr,int2e,int1e,index);
	 // (Qqr^LC1*Bqr^R)^H
	 auto qt3h = wt*oper_kernel_IOwf("cr",wf2,op2,0,dagger);
	 qt3h = qt3h.merge_cr().split_lc(wf.qrow,wf.qmid,wf.dpt_lc1().second);
	 Hwf1 += oper_compxwf_opQ("lc",qt3h,lqops,c1qops,isym,ifkr,int2e,int1e,index,dagger);
      }
/*
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
*/
   } // ifln
   Hwf += Hwf1.split_c2r(wf.qver,wf.qcol,wf.dpt_c2r().second);
   Hwf += Hwf2.split_lc1(wf.qrow,wf.qmid,wf.dpt_lc1().second); 
   Hwf.to_array(y);
}

} // ctns

#endif
