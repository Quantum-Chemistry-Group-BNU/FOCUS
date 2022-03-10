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
   direction2 dir = {qt3.dir_row(),qt3.dir_col()};
   stensor2<Tm> qt2(qt3.info.sym, qlc, qcol, dir);
   // loop over qt2
   int blc, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,blc,bc);
      auto blk2 = qt2(blc,bc); 
      // loop over compatible l*c = lc
      auto qsym_lc = qlc.get_sym(blc);
      for(const auto& p12 : dpt.at(qsym_lc)){
	 int br = std::get<0>(p12);
	 int bm = std::get<1>(p12);
	 int ioff = std::get<2>(p12);
	 const auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
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
   direction3 dir = {qt2.dir_row(),qt2.dir_col(),qt2.dir_row()};
   stensor3<Tm> qt3(qt2.info.sym, qlx, qcol, qcx, dir);
   // loop over qt2
   int blc, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,blc,bc);
      const auto blk2 = qt2(blc,bc); 
      // loop over compatible l*c = lc
      auto qsym_lc = qlc.get_sym(blc);
      for(const auto& p12 : dpt.at(qsym_lc)){
	 int br = std::get<0>(p12);
	 int bm = std::get<1>(p12);
	 int ioff = std::get<2>(p12);
	 auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
   return qt3;
}

// psi3[l,r,c] -> psi2[l,cr]
template <typename Tm> 
stensor2<Tm> merge_qt3_qt2_cr(const stensor3<Tm>& qt3,
			      const qbond& qcr, 
			      const qdpt& dpt){
   const auto& qrow = qt3.info.qrow;
   assert(qt3.dir_mid() == qt3.dir_col());
   direction2 dir = {qt3.dir_row(),qt3.dir_mid()};
   stensor2<Tm> qt2(qt3.info.sym, qrow, qcr, dir);
   // loop over qt2
   int br, bcr;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,br,bcr);
      auto blk2 = qt2(br,bcr);
      // loop over compatible c*r = cr
      auto qsym_cr = qcr.get_sym(bcr);
      for(const auto& p12 : dpt.at(qsym_cr)){
	 int bm = std::get<0>(p12);
	 int bc = std::get<1>(p12);
	 int ioff = std::get<2>(p12); 
	 const auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
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
   direction3 dir = {qt2.dir_row(),qt2.dir_col(),qt2.dir_col()};
   stensor3<Tm> qt3(qt2.info.sym, qrow, qrx, qcx, dir);
   // loop over qt2
   int br, bcr;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,br,bcr);
      const auto blk2 = qt2(br,bcr);
      // loop over compatible c*r = cr
      auto qsym_cr = qcr.get_sym(bcr);
      for(const auto& p12 : dpt.at(qsym_cr)){
	 int bm = std::get<0>(p12);
	 int bc = std::get<1>(p12);
	 int ioff = std::get<2>(p12); 
	 auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
   return qt3;
}

// psi3[l,r,c] -> psi2[lr,c]
template <typename Tm>
stensor2<Tm> merge_qt3_qt2_lr(const stensor3<Tm>& qt3,
			      const qbond& qlr, 
			      const qdpt& dpt){
   const auto& qmid = qt3.info.qmid;
   assert(qt3.dir_row() == qt3.dir_col()); 
   direction2 dir = {qt3.dir_row(),qt3.dir_mid()};
   stensor2<Tm> qt2(qt3.info.sym, qlr, qmid, dir);
   // loop over qt2
   int blr, bm;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,blr,bm);
      auto blk2 = qt2(blr,bm);
      // loop over comptabile l*r = lr
      auto qsym_lr = qlr.get_sym(blr);
      for(const auto& p12 : dpt.at(qsym_lr)){
	 int br = std::get<0>(p12);
	 int bc = std::get<1>(p12);
	 int ioff = std::get<2>(p12);
	 const auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
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
   direction3 dir = {qt2.dir_row(),qt2.dir_row(),qt2.dir_col()};
   stensor3<Tm> qt3(qt2.info.sym, qlx, qrx, qmid, dir);
   // loop over qt2
   int blr, bm;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,blr,bm);
      const auto blk2 = qt2(blr,bm);
      // loop over comptabile l*r = lr
      auto qsym_lr = qlr.get_sym(blr);
      for(const auto& p12 : dpt.at(qsym_lr)){
	 int br = std::get<0>(p12);
	 int bc = std::get<1>(p12);
	 int ioff = std::get<2>(p12);
	 auto blk3 = qt3(br,bc,bm);
	 if(blk3.empty()) continue;
	 int rdim = blk3.dim0;
	 int cdim = blk3.dim1;
	 int mdim = blk3.dim2;
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
   } // i
   return qt3;
}

} // ctns

#endif
