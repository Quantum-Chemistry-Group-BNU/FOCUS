#ifndef CONTRACT_QT3_QT3_H
#define CONTRACT_QT3_QT3_H

namespace ctns{

// --- contract_qt3_qt3 ---
template <typename Tm>
stensor2<Tm> contract_qt3_qt3(const std::string superblock,
		 	      const stensor3<Tm>& qt3a, 
			      const stensor3<Tm>& qt3b){
   stensor2<Tm> qt2;
   if(superblock == "lc"){
      qt2 = contract_qt3_qt3_lc(qt3a, qt3b);
   }else if(superblock == "cr"){
      qt2 = contract_qt3_qt3_cr(qt3a, qt3b);
   }else if(superblock == "lr"){
      qt2 = contract_qt3_qt3_lr(qt3a, qt3b);
   }else{
      std::cout << "error: no such case in contract_qt3_qt3! superblock=" 
	        << superblock << std::endl;
      exit(1);
   }
   return qt2;
}

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lc(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qmid == qt3b.info.qmid);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qcol, qt3b.info.qcol); 
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
	 auto& blk2 = qt2(br,bc);
	 if(blk2.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt3a.rows(); bx++){
            for(int bm=0; bm<qt3a.mids(); bm++){
	       const auto& blk3a = qt3a(bx,br,bm);
	       const auto& blk3b = qt3b(bx,bc,bm);
	       if(blk3a.size() == 0 || blk3b.size() == 0) continue;
	       int mdim = qt3a.info.qmid.get_dim(bm);
               for(int im=0; im<mdim; im++){
		  xgemm("C","N",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	       } // im
	    } // bm
	 } // bx
      } // bc
   } // br
   return qt2;
}

//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^* Bn^T [conjugation is taken on qt3a!]
//          c--*--/ qt3b
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_cr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir  == qt3b.info.dir); // bra dir fliped implicitly
   assert(qt3a.info.qmid == qt3b.info.qmid);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qrow, qt3b.info.qrow);
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk2 = qt2(br,bc);
	 if(blk2.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt3a.cols(); bx++){
	    for(int bm=0; bm<qt3a.mids(); bm++){
	       const auto& blk3a = qt3a(br,bx,bm);
	       const auto& blk3b = qt3b(bc,bx,bm);
	       if(blk3a.size() == 0 || blk3b.size() == 0) continue;
	       int mdim = qt3a.info.qmid.get_dim(bm);
	       for(int im=0; im<mdim; im++){
		  xgemm("N","C",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	       } // im	       
	    } // bm
	 } // bx
	 blk2.conjugate();
      } // bc
   } // br
   return qt2;
}

// 	      r|
//          /--*--\ qt3a
// q(r,c) = |x    |y	  = <r|c> = tr(A[r]^* B[c]^T)
//          \--*--/ qt3b
//            c|
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qmid, qt3b.info.qmid);
   // loop over contracted indices
   for(int bx=0; bx<qt3a.rows(); bx++){
      for(int by=0; by<qt3a.cols(); by++){
         // loop over external indices
         for(int br=0; br<qt2.rows(); br++){
	    const auto& blk3a = qt3a(bx,by,br);
	    if(blk3a.size() == 0) continue;
            for(int bc=0; bc<qt2.cols(); bc++){
               auto& blk2 = qt2(br,bc);
	       const auto& blk3b = qt3b(bx,by,bc);
	       if(blk2.size() == 0 || blk3b.size() == 0) continue;
	       int cdim = qt2.info.qcol.get_dim(bc);
	       int rdim = qt2.info.qrow.get_dim(br);
	       for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
		     auto tmp = xgemm("N","C",blk3a.get(ir),blk3b.get(ic));
		     blk2(ir,ic) += tools::conjugate(tmp.trace());
		  } // ir 
	       } // ic
            } // bc
         } // br
      } // by
   } // bx
   return qt2;
}

} // ctns

#endif
