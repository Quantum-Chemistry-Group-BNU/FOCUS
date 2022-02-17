#ifndef CONTRACT_QT2_QT2_H
#define CONTRACT_QT2_QT2_H

namespace ctns{

// --- contract_qt2_qt2 ---
// xgemm : C[i,k] = A[i,j] B[j,k]
template <typename Tm>
stensor2<Tm> contract_qt2_qt2(const stensor2<Tm>& qt2a, 
			      const stensor2<Tm>& qt2b){
   assert(qt2a.dir_col() == !qt2b.dir_row());
   assert(qt2a.info.qcol == qt2b.info.qrow);
   qsym sym = qt2a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt2a.dir_row(), qt2b.dir_col()};
   stensor2<Tm> qt2(sym, qt2a.info.qrow, qt2b.info.qcol, dir); 
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk2 = qt2(br,bc);
	 if(blk2.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt2a.cols(); bx++){
	    const auto& blk2a = qt2a(br,bx);
	    const auto& blk2b = qt2b(bx,bc);
	    if(blk2a.size() == 0 || blk2b.size() == 0) continue;
	    xgemm("N","N",1.0,blk2a,blk2b,1.0,blk2);
	 } // bx
      } // bc
   } // br
   return qt2;
}

} // ctns

#endif
