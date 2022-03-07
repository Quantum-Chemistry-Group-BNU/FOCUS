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
   for(const auto& pr : qt2.info._qblocks){
      const auto& key = pr.first;
      int br = std::get<0>(key);
      int bc = std::get<1>(key);
      auto blk2 = qt2(br,bc);
      // loop over contracted indices
      for(int bx=0; bx<qt2a.cols(); bx++){
	 if(qt2a.ifNotExist(br,bx) || qt2b.ifNotExist(bx,bc)) continue;
	 const auto blk2a = qt2a(br,bx);
	 const auto blk2b = qt2b(bx,bc);
	 xgemm("N","N",1.0,blk2a,blk2b,1.0,blk2);
      } // bx
   } // i
   return qt2;
}

} // ctns

#endif
