#ifndef RESHAPE_QT4_QT3_H
#define RESHAPE_QT4_QT3_H

namespace ctns{

// --- two-dot wavefunction: qt4[l,r,c1,c2] <-> qt3 ---
// merge_lc1: wf4[l,r,c1,c2] => wf3[lc1,r,c2]
// merge_c2r: wf4[l,r,c1,c2] => wf3[l,c2r,c1]
// merge_c1c2: wf4[l,r,c1,c2] => wf3[l,r,c1c2]

// psi4[l,r,c1,c2] -> psi3[lc1,r,c2]
template <typename Tm>
stensor3<Tm> merge_qt4_qt3_lc1(const stensor4<Tm>& qt4,
			       const qbond& qlc1, 
			       const qdpt& dpt){
   const auto& sym = qt4.info.sym;
   const auto& qver = qt4.info.qver;
   const auto& qcol = qt4.info.qcol;
   std::vector<bool> dir = {1,1,1};
   stensor3<Tm> qt3(sym, qlc1, qcol, qver, dir);
   for(int bv=0; bv<qver.size(); bv++){
      int vdim = qver.get_dim(bv);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
         for(int blc1=0; blc1<qlc1.size(); blc1++){
	    auto& blk3 = qt3(blc1,bc,bv);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*c1 = lc1
	    auto qsym_lc1 = qlc1.get_sym(blc1);
	    for(const auto& p12 : dpt.at(qsym_lc1)){
	       int br = std::get<0>(p12);
	       int bm = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.info.qrow.get_dim(br);
	       int mdim = qt4.info.qmid.get_dim(bm); 
	       const auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue; 
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	            for(int ic=0; ic<cdim; ic++){
	               for(int ir=0; ir<rdim; ir++){
	                  // psi4[l,r,c1,c2] <-> psi3[l(fast)c1,r,c2]
	                  int irm = ioff+im*rdim+ir; 
	                  blk3(irm,ic,iv) = blk4(ir,ic,im,iv); // internally c1[fast]c2
	               } // ir
	            } // ic
	          } // im
	       } // iv
	    } // p12
	 } // bc	 
      } // blc
   } // bv
   return qt3;
}
// psi4[l,r,c1,c2] <- psi3[lc1,r,c2]
template <typename Tm>
stensor4<Tm> split_qt4_qt3_lc1(const stensor3<Tm>& qt3,
			       const qbond& qlx,
			       const qbond& qc1, 
			       const qdpt& dpt){
   const auto& sym = qt3.info.sym;
   const auto& qlc1 = qt3.info.qrow;
   const auto& qver = qt3.info.qmid;
   const auto& qcol = qt3.info.qcol;
   stensor4<Tm> qt4(sym, qlx, qcol, qc1, qver);
   for(int bv=0; bv<qver.size(); bv++){
      int vdim = qver.get_dim(bv);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
         for(int blc1=0; blc1<qlc1.size(); blc1++){
	    const auto& blk3 = qt3(blc1,bc,bv);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*c = lc
	    auto qsym_lc1 = qlc1.get_sym(blc1);
	    for(const auto& p12 : dpt.at(qsym_lc1)){
	       int br = std::get<0>(p12);
	       int bm = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.info.qrow.get_dim(br);
	       int mdim = qt4.info.qmid.get_dim(bm); 
	       auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue; 
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	            for(int ic=0; ic<cdim; ic++){
	               for(int ir=0; ir<rdim; ir++){
	                  // psi4[l,r,c1,c2] <-> psi3[lc1,r,c2]
	                  int irm = ioff+im*rdim+ir; // store (ir,im) 
	                  blk4(ir,ic,im,iv) = blk3(irm,ic,iv); // internally c1[fast]c2
	               } // ir
	            } // ic
	          } // im
	       } // iv
	    } // p12
	 } // bc	 
      } // blc
   } // bv
   return qt4;
}

// psi4[l,r,c1,c2] -> psi3[l,c2r,c1]
template <typename Tm>
stensor3<Tm> merge_qt4_qt3_c2r(const stensor4<Tm>& qt4,
			       const qbond& qc2r, 
			       const qdpt& dpt){
   const auto& sym = qt4.info.sym;
   const auto& qrow = qt4.info.qrow; 
   const auto& qmid = qt4.info.qmid;
   std::vector<bool> dir = {1,1,1};
   stensor3<Tm> qt3(sym, qrow, qc2r, qmid, dir);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int br=0; br<qrow.size(); br++){
         int rdim = qrow.get_dim(br);
         for(int bc2r=0; bc2r<qc2r.size(); bc2r++){
	    auto& blk3 = qt3(br,bc2r,bm);
	    if(blk3.size() == 0) continue;
	    // loop over compatible c2*r = c2r
	    auto qsym_c2r = qc2r.get_sym(bc2r);
	    for(const auto& p12 : dpt.at(qsym_c2r)){
	       int bv = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int vdim = qt4.info.qver.get_dim(bv);
	       int cdim = qt4.info.qcol.get_dim(bc);
	       const auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	             for(int ic=0; ic<cdim; ic++){
		        for(int ir=0; ir<rdim; ir++){
	                   // psi4[l,r,c1,c2] <-> psi3[l,c2r,c1]
			   int ivc = ioff+ic*vdim+iv;
		           blk3(ir,ivc,im) = blk4(ir,ic,im,iv);
		        } // ir
		     } // ic
	          } // im
	       } // iv
	    } // p12
	 } // bc2r
      } // br
   } // bm
   return qt3;
}
// psi4[l,r,c1,c2] <- psi3[l,c2r,c1]
template <typename Tm>
stensor4<Tm> split_qt4_qt3_c2r(const stensor3<Tm>& qt3,
			       const qbond& qc2,
			       const qbond& qrx, 
			       const qdpt& dpt){
   const auto& sym = qt3.info.sym;
   const auto& qrow = qt3.info.qrow; 
   const auto& qmid = qt3.info.qmid;
   const auto& qc2r = qt3.info.qcol;
   stensor4<Tm> qt4(sym, qrow, qrx, qmid, qc2);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int br=0; br<qrow.size(); br++){
         int rdim = qrow.get_dim(br);
         for(int bc2r=0; bc2r<qc2r.size(); bc2r++){
	    const auto& blk3 = qt3(br,bc2r,bm);
	    if(blk3.size() == 0) continue;
	    // loop over compatible c2*r = c2r
	    auto qsym_c2r = qc2r.get_sym(bc2r);
	    for(const auto& p12 : dpt.at(qsym_c2r)){
	       int bv = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int vdim = qt4.info.qver.get_dim(bv);
	       int cdim = qt4.info.qcol.get_dim(bc);
	       auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	             for(int ic=0; ic<cdim; ic++){
		        for(int ir=0; ir<rdim; ir++){
	                   // psi4[l,r,c1,c2] <-> psi3[l,c2r,c1]
			   int ivc = ioff+ic*vdim+iv;
		           blk4(ir,ic,im,iv) = blk3(ir,ivc,im);
		        } // ir
		     } // ic
	          } // im
	       } // iv
	    } // p12
	 } // bc2r
      } // br
   } // bm
   return qt4;
}

// psi4[l,r,c1,c2] -> psi3[l,r,c1c2]
template <typename Tm>
stensor3<Tm> merge_qt4_qt3_c1c2(const stensor4<Tm>& qt4,
			        const qbond& qc1c2,
				const qdpt& dpt){
   const auto& sym = qt4.info.sym;
   const auto& qrow = qt4.info.qrow;
   const auto& qcol = qt4.info.qcol;
   std::vector<bool> dir = {1,1,1};
   stensor3<Tm> qt3(sym, qrow, qcol, qc1c2, dir);
   for(int bc1c2=0; bc1c2<qc1c2.size(); bc1c2++){
      for(int br=0; br<qrow.size(); br++){
	 int rdim = qrow.get_dim(br);
         for(int bc=0; bc<qcol.size(); bc++){
	    int cdim = qcol.get_dim(bc);
	    auto& blk3 = qt3(br,bc,bc1c2);
	    if(blk3.size() == 0) continue;
	    // lover over compatible c1*c2 = c1c2
	    auto qsym_c1c2 = qc1c2.get_sym(bc1c2);
	    for(const auto& p12 : dpt.at(qsym_c1c2)){
	       int bm = std::get<0>(p12);
	       int bv = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int mdim = qt4.info.qmid.get_dim(bm);
	       int vdim = qt4.info.qver.get_dim(bv);
	       const auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
	       		   // psi4[l,r,c1,c2] <-> psi3[l,r,c1c2]
		           int imv = ioff+iv*mdim+im; 
			   blk3(ir,ic,imv) = blk4(ir,ic,im,iv);
			} // ir
		     } // ic
		  } // im
	       } // iv
	    } // p12
	 } // bc
      } // br
   } // bc1c2
   return qt3;
}
// psi4[l,r,c1,c2] <- psi3[l,r,c1c2]
template <typename Tm>
stensor4<Tm> split_qt4_qt3_c1c2(const stensor3<Tm>& qt3,
			        const qbond& qc1,
				const qbond& qc2,
				const qdpt& dpt){
   const auto& sym = qt3.info.sym;
   const auto& qc1c2 = qt3.info.qmid;
   const auto& qrow = qt3.info.qrow;
   const auto& qcol = qt3.info.qcol;
   stensor4<Tm> qt4(sym, qrow, qcol, qc1, qc2);
   for(int bc1c2=0; bc1c2<qc1c2.size(); bc1c2++){
      for(int br=0; br<qrow.size(); br++){
	 int rdim = qrow.get_dim(br);
         for(int bc=0; bc<qcol.size(); bc++){
	    int cdim = qcol.get_dim(bc);
	    const auto& blk3 = qt3(br,bc,bc1c2);
	    if(blk3.size() == 0) continue;
	    // lover over compatible c1*c2 = c1c2
	    auto qsym_c1c2 = qc1c2.get_sym(bc1c2);
	    for(const auto& p12 : dpt.at(qsym_c1c2)){
	       int bm = std::get<0>(p12);
	       int bv = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int mdim = qt4.info.qmid.get_dim(bm);
	       int vdim = qt4.info.qver.get_dim(bv);
	       auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
	       		   // psi4[l,r,c1,c2] <-> psi3[l,r,c1c2]
		           int imv = ioff+iv*mdim+im; 
			   blk4(ir,ic,im,iv) = blk3(ir,ic,imv);
			} // ir
		     } // ic
		  } // im
	       } // iv
	    } // p12
	 } // bc
      } // br
   } // bc1c2
   return qt4;
}

} // ctns

#endif
