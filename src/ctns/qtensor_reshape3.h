#ifndef QTENSOR_RESHAPE3_H
#define QTENSOR_RESHAPE3_H

#include "qtensor.h"
#include "ctns_qdpt.h"

namespace ctns{

// --- one-dot wavefunction: qt3[l,c,r] <-> qt2 ---

// psi3[l,c,r] <-> psi2[lc,r]
template <typename Tm> 
qtensor2<Tm> merge_qt3_qt2_lc(const qtensor3<Tm>& qt3,
			      const qbond& qlc,
			      const qdpt& dpt){
   const auto& qcol = qt3.qcol;
   assert(qt3.dir[0] == qt3.dir[1]); // dc == dl
   std::vector<bool> dir = {qt3.dir[1],qt3.dir[2]};
   qtensor2<Tm> qt2(qt3.sym, qlc, qcol, dir);
   for(int bc=0; bc<qt3.cols(); bc++){
      int cdim = qt3.qcol.get_dim(bc);
      for(int blc=0; blc<qlc.size(); blc++){
	 auto& blk2 = qt2(blc,bc);
	 if(blk2.size() == 0) continue;
	 // loop over compatible l*c = lc
	 auto qsym_lc = qlc.get_sym(blc);
         for(const auto& p12 : dpt.at(qsym_lc)){
	    int br = std::get<0>(p12);
	    int bm = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
	    int rdim = qt3.qrow.get_dim(br);
	    int mdim = qt3.qmid.get_dim(bm);
	    const auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	    for(int im=0; im<mdim; im++){
	       for(int ic=0; ic<cdim; ic++){
	          for(int ir=0; ir<rdim; ir++){
		     // qt3[l,c,r] -> qt2[lc,r]: storage l[fast]*c!
	             int irm = ioff+im*rdim+ir;
	             blk2(irm,ic) = blk3[im](ir,ic);
	          } // ir
	       } // ic
	    } // im
         } // p12			 
      } // blc
   } // bc
   return qt2;
}
template <typename Tm>
qtensor3<Tm> split_qt3_qt2_lc(const qtensor2<Tm>& qt2,
			      const qbond& qlx,
			      const qbond& qcx,
			      const qdpt& dpt){
   const auto& qlc  = qt2.qrow; 
   const auto& qcol = qt2.qcol;
   std::vector<bool> dir = {qt2.dir[0],qt2.dir[0],qt2.dir[1]};
   qtensor3<Tm> qt3(qt2.sym, qcx, qlx, qcol, dir);
   for(int bc=0; bc<qt3.cols(); bc++){
      int cdim = qt3.qcol.get_dim(bc);
      for(int blc=0; blc<qlc.size(); blc++){
	 const auto& blk2 = qt2(blc,bc);
	 if(blk2.size() == 0) continue;
	 // loop over compatible l*c = lc
	 auto qsym_lc = qlc.get_sym(blc);
         for(const auto& p12 : dpt.at(qsym_lc)){
	    int br = std::get<0>(p12);
	    int bm = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
	    int rdim = qt3.qrow.get_dim(br);
	    int mdim = qt3.qmid.get_dim(bm);
	    auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	    for(int im=0; im<mdim; im++){
	       for(int ic=0; ic<cdim; ic++){
	          for(int ir=0; ir<rdim; ir++){
		     // qt2[lc,r] -> qt3[l,c,r], storage l[fast]c
	             int irm = ioff+im*rdim+ir;
	             blk3[im](ir,ic) = blk2(irm,ic); 
	          } // ir
	       } // ic
	    } // im
         } // p12			 
      } // blc
   } // bc
   return qt3;
}

// psi3[l,c,r] <-> psi2[l,cr]
template <typename Tm> 
qtensor2<Tm> merge_qt3_qt2_cr(const qtensor3<Tm>& qt3,
			      const qbond& qcr, 
			      const qdpt& dpt){
   const auto& qrow = qt3.qrow;
   assert(qt3.dir[0] == qt3.dir[2]); // dc == dr
   std::vector<bool> dir = {qt3.dir[1],qt3.dir[2]};
   qtensor2<Tm> qt2(qt3.sym, qrow, qcr, dir);
   for(int br=0; br<qt3.rows(); br++){
      int rdim = qt3.qrow.get_dim(br);
      for(int bcr=0; bcr<qcr.size(); bcr++){
	 auto& blk2 = qt2(br,bcr);
	 if(blk2.size() == 0) continue;
	 // loop over compatible c*r = cr
	 auto qsym_cr = qcr.get_sym(bcr);
	 for(const auto& p12 : dpt.at(qsym_cr)){
	    int bm = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12); 
	    int mdim = qt3.qmid.get_dim(bm);
	    int cdim = qt3.qcol.get_dim(bc);
	    const auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	    for(int ic=0; ic<cdim; ic++){
	       for(int im=0; im<mdim; im++){
		  // qt3[l,c,r] -> qt2[l,cr], storage c[fast]r
	          int imc = ioff+ic*mdim+im;
	          for(int ir=0; ir<rdim; ir++){
	             blk2(ir,imc) = blk3[im](ir,ic); 
	          } // ir
	       } // ic
	    } // im
	 } // p12 
      } // bcr 
   } // br
   return qt2;
}
template <typename Tm> 
qtensor3<Tm> split_qt3_qt2_cr(const qtensor2<Tm>& qt2,
			      const qbond& qcx,
			      const qbond& qrx,
			      const qdpt& dpt){
   const auto& qrow = qt2.qrow;
   const auto& qcr  = qt2.qcol;
   std::vector<bool> dir = {qt2.dir[1],qt2.dir[0],qt2.dir[1]};
   qtensor3<Tm> qt3(qt2.sym, qcx, qrow, qrx, dir);
   for(int br=0; br<qt3.rows(); br++){
      int rdim = qt3.qrow.get_dim(br);
      for(int bcr=0; bcr<qcr.size(); bcr++){
	 const auto& blk2 = qt2(br,bcr);
	 if(blk2.size() == 0) continue;
	 // loop over compatible c*r = cr
	 auto qsym_cr = qcr.get_sym(bcr);
	 for(const auto& p12 : dpt.at(qsym_cr)){
	    int bm = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12); 
	    int mdim = qt3.qmid.get_dim(bm);
	    int cdim = qt3.qcol.get_dim(bc);
	    auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	    for(int ic=0; ic<cdim; ic++){
	       for(int im=0; im<mdim; im++){
		  // qt2[l,cr] -> qt3[l,c,r], storage c[fast]r
	          int imc = ioff+ic*mdim+im;
	          for(int ir=0; ir<rdim; ir++){
		     blk3[im](ir,ic) = blk2(ir,imc); 
	          } // ir
	       } // ic
	    } // im
	 } // p12 
      } // bcr 
   } // br
   return qt3;
}

// psi3[l,c,r] <-> psi2[lr,c]
template <typename Tm>
qtensor2<Tm> merge_qt3_qt2_lr(const qtensor3<Tm>& qt3,
			      const qbond& qlr, 
			      const qdpt& dpt){
   const auto& qmid = qt3.qmid;
   assert(qt3.dir[1] == qt3.dir[2]); // dl == dr
   std::vector<bool> dir = {qt3.dir[1],qt3.dir[0]};
   qtensor2<Tm> qt2(qt3.sym, qlr, qmid, dir);
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.qmid.get_dim(bm);
      for(int blr=0; blr<qlr.size(); blr++){
	 auto& blk2 = qt2(blr,bm);
	 if(blk2.size() == 0) continue;
	 // loop over comptabile l*r = lr
	 auto qsym_lr = qlr.get_sym(blr);
	 for(const auto& p12 : dpt.at(qsym_lr)){
	    int br = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
            int rdim = qt3.qrow.get_dim(br);
	    int cdim = qt3.qcol.get_dim(bc);
	    const auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
			// qt3[l,c,r] -> qt2[lr,c], storage l[fast]r
		        int irc = ioff+ic*rdim+ir;
		        blk2(irc,im) = blk3[im](ir,ic);
		     } // ir
		  } // ic
	       } // im
	 } // p12
      } // blr
   } // bm
   return qt2;
}
template <typename Tm>
qtensor3<Tm> split_qt3_qt2_lr(const qtensor2<Tm>& qt2,
			      const qbond& qlx,
			      const qbond& qrx,
			      const qdpt& dpt){
   const auto& qlr  = qt2.qrow;
   const auto& qmid = qt2.qcol;
   std::vector<bool> dir = {qt2.dir[1],qt2.dir[0],qt2.dir[0]};
   qtensor3<Tm> qt3(qt2.sym, qmid, qlx, qrx, dir);
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.qmid.get_dim(bm);
      for(int blr=0; blr<qlr.size(); blr++){
	 const auto& blk2 = qt2(blr,bm);
	 if(blk2.size() == 0) continue;
	 // loop over comptabile l*r = lr
	 auto qsym_lr = qlr.get_sym(blr);
	 for(const auto& p12 : dpt.at(qsym_lr)){
	    int br = std::get<0>(p12);
	    int bc = std::get<1>(p12);
	    int ioff = std::get<2>(p12);
            int rdim = qt3.qrow.get_dim(br);
	    int cdim = qt3.qcol.get_dim(bc);
	    auto& blk3 = qt3(bm,br,bc);
	    if(blk3.size() == 0) continue;
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
			// qt3[l,c,r] -> qt2[lr,c], storage l[fast]r
		        int irc = ioff+ic*rdim+ir;
		       	blk3[im](ir,ic) = blk2(irc,im);
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
