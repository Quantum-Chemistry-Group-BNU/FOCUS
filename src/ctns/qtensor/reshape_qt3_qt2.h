#ifndef RESHAPE_QT3_QT2_H
#define RESHAPE_QT3_QT2_H

namespace ctns{

// --- one-dot wavefunction: qt3[l,r,r] <-> qt2 ---
// merge_lc: psi3[l,r,c] => psi2[lc,r]
// merge_cr: psi3[l,r,c] => psi2[l,cr]
// merge_lr: psi3[l,r,c] => psi2[lr,c]

// psi3[l,r,c] -> psi2[lc,r]
template <typename Tm> 
stensor2<Tm> merge_qt3_qt2_lc(const stensor3<Tm>& qt3,
			      const qbond& qlc,
			      const qdpt& dpt){
   const auto& qcol = qt3.info.qcol;
   // dl == dc: only merge dimensions with the same direction
   assert(qt3.dir_row() == qt3.dir_mid()); 
   std::vector<bool> dir = {qt3.dir_row(),qt3.dir_col()};
   stensor2<Tm> qt2(qt3.info.sym, qlc, qcol, dir);
   for(int bc=0; bc<qt3.cols(); bc++){
      int cdim = qt3.info.qcol.get_dim(bc);
      for(int blc=0; blc<qlc.size(); blc++){
	 auto& blk2 = qt2(blc,bc);
	 if(blk2.size() == 0) continue;
	 // loop over compatible l*c = lc
	 auto qsym_lc = qlc.get_sym(blc);
         for(const auto& p12 : dpt.at(qsym_lc)){
	    int br = std::get<0>(p12);
	    int bm = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
	    int rdim = qt3.info.qrow.get_dim(br);
	    int mdim = qt3.info.qmid.get_dim(bm);
	    const auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    for(int im=0; im<mdim; im++){
	       for(int ic=0; ic<cdim; ic++){
	          for(int ir=0; ir<rdim; ir++){
		     // qt3[l,r,c] -> qt2[lc,r]: storage l[fast]*c!
	             int irm = ioff+im*rdim+ir;
	             blk2(irm,ic) = blk3(ir,ic,im);
	          } // ir
	       } // ic
	    } // im
         } // p12			 
      } // blc
   } // bc
   return qt2;
}
// psi3[l,r,c] <- psi2[lc,r]
template <typename Tm>
stensor3<Tm> split_qt3_qt2_lc(const stensor2<Tm>& qt2,
			      const qbond& qlx,
			      const qbond& qcx,
			      const qdpt& dpt){
   const auto& qlc  = qt2.info.qrow; 
   const auto& qcol = qt2.info.qcol;
   std::vector<bool> dir = {qt2.dir_row(),qt2.dir_col(),qt2.dir_row()};
   stensor3<Tm> qt3(qt2.info.sym, qlx, qcol, qcx, dir);
   for(int bc=0; bc<qt3.cols(); bc++){
      int cdim = qt3.info.qcol.get_dim(bc);
      for(int blc=0; blc<qlc.size(); blc++){
	 const auto& blk2 = qt2(blc,bc);
	 if(blk2.size() == 0) continue;
	 // loop over compatible l*c = lc
	 auto qsym_lc = qlc.get_sym(blc);
         for(const auto& p12 : dpt.at(qsym_lc)){
	    int br = std::get<0>(p12);
	    int bm = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
	    int rdim = qt3.info.qrow.get_dim(br);
	    int mdim = qt3.info.qmid.get_dim(bm);
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    for(int im=0; im<mdim; im++){
	       for(int ic=0; ic<cdim; ic++){
	          for(int ir=0; ir<rdim; ir++){
		     // qt2[lc,r] -> qt3[l,r,c], storage l[fast]c
	             int irm = ioff+im*rdim+ir;
	             blk3(ir,ic,im) = blk2(irm,ic); 
	          } // ir
	       } // ic
	    } // im
         } // p12			 
      } // blc
   } // bc
   return qt3;
}

// psi3[l,r,c] -> psi2[l,cr]
template <typename Tm> 
stensor2<Tm> merge_qt3_qt2_cr(const stensor3<Tm>& qt3,
			      const qbond& qcr, 
			      const qdpt& dpt){
   const auto& qrow = qt3.info.qrow;
   assert(qt3.dir_mid() == qt3.dir_col());
   std::vector<bool> dir = {qt3.dir_row(),qt3.dir_mid()};
   stensor2<Tm> qt2(qt3.info.sym, qrow, qcr, dir);
   for(int br=0; br<qt3.rows(); br++){
      int rdim = qt3.info.qrow.get_dim(br);
      for(int bcr=0; bcr<qcr.size(); bcr++){
	 auto& blk2 = qt2(br,bcr);
	 if(blk2.size() == 0) continue;
	 // loop over compatible c*r = cr
	 auto qsym_cr = qcr.get_sym(bcr);
	 for(const auto& p12 : dpt.at(qsym_cr)){
	    int bm = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12); 
	    int mdim = qt3.info.qmid.get_dim(bm);
	    int cdim = qt3.info.qcol.get_dim(bc);
	    const auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    for(int ic=0; ic<cdim; ic++){
	       for(int im=0; im<mdim; im++){
		  // qt3[l,r,c] -> qt2[l,cr], storage c[fast]r
	          int imc = ioff+ic*mdim+im;
	          for(int ir=0; ir<rdim; ir++){
	             blk2(ir,imc) = blk3(ir,ic,im); 
	          } // ir
	       } // ic
	    } // im
	 } // p12 
      } // bcr 
   } // br
   return qt2;
}
// psi3[l,r,c] <- psi2[l,cr]
template <typename Tm> 
stensor3<Tm> split_qt3_qt2_cr(const stensor2<Tm>& qt2,
			      const qbond& qcx,
			      const qbond& qrx,
			      const qdpt& dpt){
   const auto& qrow = qt2.info.qrow;
   const auto& qcr  = qt2.info.qcol;
   std::vector<bool> dir = {qt2.dir_row(),qt2.dir_col(),qt2.dir_col()};
   stensor3<Tm> qt3(qt2.info.sym, qrow, qrx, qcx, dir);
   for(int br=0; br<qt3.rows(); br++){
      int rdim = qt3.info.qrow.get_dim(br);
      for(int bcr=0; bcr<qcr.size(); bcr++){
	 const auto& blk2 = qt2(br,bcr);
	 if(blk2.size() == 0) continue;
	 // loop over compatible c*r = cr
	 auto qsym_cr = qcr.get_sym(bcr);
	 for(const auto& p12 : dpt.at(qsym_cr)){
	    int bm = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12); 
	    int mdim = qt3.info.qmid.get_dim(bm);
	    int cdim = qt3.info.qcol.get_dim(bc);
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    for(int ic=0; ic<cdim; ic++){
	       for(int im=0; im<mdim; im++){
		  // qt2[l,cr] -> qt3[l,r,c], storage c[fast]r
	          int imc = ioff+ic*mdim+im;
	          for(int ir=0; ir<rdim; ir++){
		     blk3(ir,ic,im) = blk2(ir,imc); 
	          } // ir
	       } // ic
	    } // im
	 } // p12 
      } // bcr 
   } // br
   return qt3;
}

// psi3[l,r,c] -> psi2[lr,c]
template <typename Tm>
stensor2<Tm> merge_qt3_qt2_lr(const stensor3<Tm>& qt3,
			      const qbond& qlr, 
			      const qdpt& dpt){
   const auto& qmid = qt3.info.qmid;
   assert(qt3.dir_row() == qt3.dir_col()); 
   std::vector<bool> dir = {qt3.dir_row(),qt3.dir_mid()};
   stensor2<Tm> qt2(qt3.info.sym, qlr, qmid, dir);
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.info.qmid.get_dim(bm);
      for(int blr=0; blr<qlr.size(); blr++){
	 auto& blk2 = qt2(blr,bm);
	 if(blk2.size() == 0) continue;
	 // loop over comptabile l*r = lr
	 auto qsym_lr = qlr.get_sym(blr);
	 for(const auto& p12 : dpt.at(qsym_lr)){
	    int br = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
            int rdim = qt3.info.qrow.get_dim(br);
	    int cdim = qt3.info.qcol.get_dim(bc);
	    const auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
			// qt3[l,r,c] -> qt2[lr,c], storage l[fast]r
		        int irc = ioff+ic*rdim+ir;
		        blk2(irc,im) = blk3(ir,ic,im);
		     } // ir
		  } // ic
	       } // im
	 } // p12
      } // blr
   } // bm
   return qt2;
}
// psi3[l,r,c] <- psi2[lr,c]
template <typename Tm>
stensor3<Tm> split_qt3_qt2_lr(const stensor2<Tm>& qt2,
			      const qbond& qlx,
			      const qbond& qrx,
			      const qdpt& dpt){
   const auto& qlr  = qt2.info.qrow;
   const auto& qmid = qt2.info.qcol;
   std::vector<bool> dir = {qt2.dir_row(),qt2.dir_row(),qt2.dir_col()};
   stensor3<Tm> qt3(qt2.info.sym, qlx, qrx, qmid, dir);
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.info.qmid.get_dim(bm);
      for(int blr=0; blr<qlr.size(); blr++){
	 const auto& blk2 = qt2(blr,bm);
	 if(blk2.size() == 0) continue;
	 // loop over comptabile l*r = lr
	 auto qsym_lr = qlr.get_sym(blr);
	 for(const auto& p12 : dpt.at(qsym_lr)){
	    int br = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
            int rdim = qt3.info.qrow.get_dim(br);
	    int cdim = qt3.info.qcol.get_dim(bc);
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
			// qt3[l,c,r] -> qt2[lr,c], storage l[fast]r
		        int irc = ioff+ic*rdim+ir;
		       	blk3(ir,ic,im) = blk2(irc,im);
		     } // ir
		  } // ic
	       } // im
	 } // p12
      } // blr
   } // bm
   return qt3;
}

} // ctns

#endif
