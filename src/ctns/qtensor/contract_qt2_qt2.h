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
   direction2 dir = {qt2a.dir_row(), qt2b.dir_col()};
   stensor2<Tm> qt2(sym, qt2a.info.qrow, qt2b.info.qcol, dir); 
   // loop over external indices
   int br, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int idx = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(idx,br,bc);
      auto blk2 = qt2(br,bc);
      // loop over contracted indices
      for(int bx=0; bx<qt2a.cols(); bx++){
	 const auto blk2a = qt2a(br,bx);
	 const auto blk2b = qt2b(bx,bc);
	 if(blk2a.empty() || blk2b.empty()) continue;
	 xgemm("N","N",1.0,blk2a,blk2b,1.0,blk2);
      } // bx
   } // i
   return qt2;
}

} // ctns

#endif
