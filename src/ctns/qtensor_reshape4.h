#ifndef QTENSOR_RESHAPE4_H
#define QTENSOR_RESHAPE4_H

#include "qtensor.h"
#include "ctns_qdpt.h"

namespace ctns{

// --- two-dot wavefunction: qt4[l,c1,c2,r] <-> qt3 ---

// psi4[l,c1,c2,r] <-> psi3[lc1,c2,r]
template <typename Tm>
qtensor3<Tm> merge_qt4_qt3_lc1(const qtensor4<Tm>& qt4,
			       const qbond& qlc1, 
			       const qdpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qver = qt4.qver;
   const auto& qcol = qt4.qcol;
   std::vector<bool> dir = {1,1,1};
   qtensor3<Tm> qt3(sym, qver, qlc1, qcol, dir);
   for(int bv=0; bv<qver.size(); bv++){
      int vdim = qver.get_dim(bv);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
         for(int blc1=0; blc1<qlc1.size(); blc1++){
	    auto& blk3 = qt3(bv,blc1,bc);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*c1 = lc1
	    auto qsym_lc1 = qlc1.get_sym(blc1);
	    for(const auto& p12 : dpt.at(qsym_lc1)){
	       int br = std::get<0>(p12);
	       int bm = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.qrow.get_dim(br);
	       int mdim = qt4.qmid.get_dim(bm); 
	       const auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue; 
	       // psi4[l,c1,c2,r] <-> psi3[lc1,c2,r]
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	            for(int ic=0; ic<cdim; ic++){
	               for(int ir=0; ir<rdim; ir++){
	                  int ilc = ioff+im*rdim+ir; // store (ir,im) 
	                  blk3[iv](ilc,ic) = blk4[iv*mdim+im](ir,ic); // internally c1[fast]c2
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
// psi4[l,c1,c2,r] <-> psi3[lc1,c2,r]
template <typename Tm>
qtensor4<Tm> split_qt4_qt3_lc1(const qtensor3<Tm>& qt3,
			       const qbond& qlx,
			       const qbond& qc1, 
			       const qdpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qlc1 = qt3.qrow;
   const auto& qver = qt3.qmid;
   const auto& qcol = qt3.qcol;
   qtensor4<Tm> qt4(sym, qc1, qver, qlx, qcol);
   for(int bv=0; bv<qver.size(); bv++){
      int vdim = qver.get_dim(bv);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
         for(int blc1=0; blc1<qlc1.size(); blc1++){
	    const auto& blk3 = qt3(bv,blc1,bc);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*c = lc
	    auto qsym_lc1 = qlc1.get_sym(blc1);
	    for(const auto& p12 : dpt.at(qsym_lc1)){
	       int br = std::get<0>(p12);
	       int bm = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.qrow.get_dim(br);
	       int mdim = qt4.qmid.get_dim(bm); 
	       auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue; 
	       // psi4[l,c1,c2,r] <-> psi3[lc1,c2,r]
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	            for(int ic=0; ic<cdim; ic++){
	               for(int ir=0; ir<rdim; ir++){
	                  int ilc = ioff+im*rdim+ir; // store (ir,im) 
	                  blk4[iv*mdim+im](ir,ic) = blk3[iv](ilc,ic); // internally c1[fast]c2
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

// psi4[l,c1,c2,r] <-> psi3[l,c1,c2r]
template <typename Tm>
qtensor3<Tm> merge_qt4_qt3_c2r(const qtensor4<Tm>& qt4,
			       const qbond& qc2r, 
			       const qdpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qrow = qt4.qrow; 
   const auto& qmid = qt4.qmid;
   std::vector<bool> dir = {1,1,1};
   qtensor3<Tm> qt3(sym, qmid, qrow, qc2r, dir);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int br=0; br<qrow.size(); br++){
         int rdim = qrow.get_dim(br);
         for(int bc2r=0; bc2r<qc2r.size(); bc2r++){
	    auto& blk3 = qt3(bm,br,bc2r);
	    if(blk3.size() == 0) continue;
	    // loop over compatible c2*r = c2r
	    auto qsym_c2r = qc2r.get_sym(bc2r);
	    for(const auto& p12 : dpt.at(qsym_c2r)){
	       int bv = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int vdim = qt4.qver.get_dim(bv);
	       int cdim = qt4.qcol.get_dim(bc);
	       const auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[l,c1,c2r]
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	             for(int ic=0; ic<cdim; ic++){
		        for(int ir=0; ir<rdim; ir++){
			   int icr = ioff+ic*vdim+iv;
		           blk3[im](ir,icr) = blk4[iv*mdim+im](ir,ic);
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
// psi4[l,c1,c2,r] <-> psi3[l,c1,c2r]
template <typename Tm>
qtensor4<Tm> split_qt4_qt3_c2r(const qtensor3<Tm>& qt3,
			       const qbond& qc2,
			       const qbond& qrx, 
			       const qdpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qrow = qt3.qrow; 
   const auto& qmid = qt3.qmid;
   const auto& qc2r = qt3.qcol;
   qtensor4<Tm> qt4(sym, qmid, qc2, qrow, qrx);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int br=0; br<qrow.size(); br++){
         int rdim = qrow.get_dim(br);
         for(int bc2r=0; bc2r<qc2r.size(); bc2r++){
	    const auto& blk3 = qt3(bm,br,bc2r);
	    if(blk3.size() == 0) continue;
	    // loop over compatible c2*r = c2r
	    auto qsym_c2r = qc2r.get_sym(bc2r);
	    for(const auto& p12 : dpt.at(qsym_c2r)){
	       int bv = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int vdim = qt4.qver.get_dim(bv);
	       int cdim = qt4.qcol.get_dim(bc);
	       auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[l,c1,c2r]
	       for(int iv=0; iv<vdim; iv++){
	          for(int im=0; im<mdim; im++){
	             for(int ic=0; ic<cdim; ic++){
		        for(int ir=0; ir<rdim; ir++){
			   int icr = ioff+ic*vdim+iv;
		           blk4[iv*mdim+im](ir,ic) = blk3[im](ir,icr);
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

// psi4[l,c1,c2,r] <-> psi3[l,c1c2,r]
template <typename Tm>
qtensor3<Tm> merge_qt4_qt3_c1c2(const qtensor4<Tm>& qt4,
			        const qbond& qc1c2,
				const qdpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qrow = qt4.qrow;
   const auto& qcol = qt4.qcol;
   std::vector<bool> dir = {1,1,1};
   qtensor3<Tm> qt3(sym, qc1c2, qrow, qcol, dir);
   for(int bc1c2=0; bc1c2<qc1c2.size(); bc1c2++){
      for(int br=0; br<qrow.size(); br++){
	 int rdim = qrow.get_dim(br);
         for(int bc=0; bc<qcol.size(); bc++){
	    int cdim = qcol.get_dim(bc);
	    auto& blk3 = qt3(bc1c2,br,bc);
	    if(blk3.size() == 0) continue;
	    // lover over compatible c1*c2 = c1c2
	    auto qsym_c1c2 = qc1c2.get_sym(bc1c2);
	    for(const auto& p12 : dpt.at(qsym_c1c2)){
	       int bm = std::get<0>(p12);
	       int bv = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int mdim = qt4.qmid.get_dim(bm);
	       int vdim = qt4.qver.get_dim(bv);
	       const auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[l,c1c2,r]
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
		           int imv = ioff+iv*mdim+im; 
			   blk3[imv](ir,ic) = blk4[iv*mdim+im](ir,ic);
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
// psi4[l,c1,c2,r] <-> psi3[l,c1c2,r]
template <typename Tm>
qtensor4<Tm> split_qt4_qt3_c1c2(const qtensor3<Tm>& qt3,
			        const qbond& qc1,
				const qbond& qc2,
				const qdpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qc1c2 = qt3.qmid;
   const auto& qrow = qt3.qrow;
   const auto& qcol = qt3.qcol;
   qtensor4<Tm> qt4(sym, qc1, qc2, qrow, qcol);
   for(int bc1c2=0; bc1c2<qc1c2.size(); bc1c2++){
      for(int br=0; br<qrow.size(); br++){
	 int rdim = qrow.get_dim(br);
         for(int bc=0; bc<qcol.size(); bc++){
	    int cdim = qcol.get_dim(bc);
	    const auto& blk3 = qt3(bc1c2,br,bc);
	    if(blk3.size() == 0) continue;
	    // lover over compatible c1*c2 = c1c2
	    auto qsym_c1c2 = qc1c2.get_sym(bc1c2);
	    for(const auto& p12 : dpt.at(qsym_c1c2)){
	       int bm = std::get<0>(p12);
	       int bv = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int mdim = qt4.qmid.get_dim(bm);
	       int vdim = qt4.qver.get_dim(bv);
	       auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[l,c1c2,r]
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
		           int imv = ioff+iv*mdim+im; 
			   blk4[iv*mdim+im](ir,ic) = blk3[imv](ir,ic);
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

// psi4[l,c1,c2,r] <-> psi3[lr,c1,c2]
template <typename Tm>
qtensor3<Tm> merge_qt4_qt3_lr(const qtensor4<Tm>& qt4,
			      const qbond& qlr,
			      const qdpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qmid = qt4.qmid;
   const auto& qver = qt4.qver;
   std::vector<bool> dir = {1,1,1};
   qtensor3<Tm> qt3(sym, qmid, qlr, qver, dir);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int bv=0; bv<qver.size(); bv++){
	 int vdim = qver.get_dim(bv);
         for(int blr=0; blr<qlr.size(); blr++){
	    auto& blk3 = qt3(bm, blr, bv);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*r = lr
	    auto qsym_lr = qlr.get_sym(blr);
	    for(const auto& p12 : dpt.at(qsym_lr)){
	       int br = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.qrow.get_dim(br);
	       int cdim = qt4.qcol.get_dim(bc);
	       const auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[lr,c1,c2]
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
			   int ilr = ioff+ic*rdim+ir;
			   blk3[im](ilr,iv) = blk4[iv*mdim+im](ir,ic);
			} // ir
		     } // ic
		  } // im
	       } // iv
	    } // p12
         } // blr
      } // bc
   } // bm
   return qt3;
}
// psi4[l,c1,c2,r] <-> psi3[lr,c1,c2]
template <typename Tm>
qtensor4<Tm> split_qt4_qt3_lr(const qtensor3<Tm>& qt3,
			      const qbond& qlx,
			      const qbond& qrx,
			      const qdpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qlr = qt3.qrow;
   const auto& qmid = qt3.qmid;
   const auto& qver = qt3.qcol;
   qtensor4<Tm> qt4(sym, qmid, qver, qlx, qrx);
   for(int bm=0; bm<qmid.size(); bm++){
      int mdim = qmid.get_dim(bm);
      for(int bv=0; bv<qver.size(); bv++){
	 int vdim = qver.get_dim(bv);
         for(int blr=0; blr<qlr.size(); blr++){
	    const auto& blk3 = qt3(bm, blr, bv);
	    if(blk3.size() == 0) continue;
	    // loop over compatible l*r = lr
	    auto qsym_lr = qlr.get_sym(blr);
	    for(const auto& p12 : dpt.at(qsym_lr)){
	       int br = std::get<0>(p12);
	       int bc = std::get<1>(p12);
	       int ioff = std::get<2>(p12);
	       int rdim = qt4.qrow.get_dim(br);
	       int cdim = qt4.qcol.get_dim(bc);
	       auto& blk4 = qt4(bm,bv,br,bc);
	       if(blk4.size() == 0) continue;
	       // psi4[l,c1,c2,r] <-> psi3[lr,c1,c2]
	       for(int iv=0; iv<vdim; iv++){
		  for(int im=0; im<mdim; im++){
		     for(int ic=0; ic<cdim; ic++){
			for(int ir=0; ir<rdim; ir++){
			   int ilr = ioff+ic*rdim+ir;
			   blk4[iv*mdim+im](ir,ic) = blk3[im](ilr,iv);
			} // ir
		     } // ic
		  } // im
	       } // iv
	    } // p12
         } // blr
      } // bc
   } // bm
   return qt4;
}

} // ctns

#endif
