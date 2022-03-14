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

template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lc(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qmid == qt3b.info.qmid);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qcol, qt3b.info.qcol); 
   contract_qt3_qt3_info_lc(qt3a.info, qt3a.data(), qt3b.info, qt3b.data(),
		            qt2.info, qt2.data());
   return qt2;
}

template <typename Tm>
stensor2<Tm> contract_qt3_qt3_cr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir  == qt3b.info.dir); // bra dir fliped implicitly
   assert(qt3a.info.qmid == qt3b.info.qmid);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qrow, qt3b.info.qrow);
   contract_qt3_qt3_info_cr(qt3a.info, qt3a.data(), qt3b.info, qt3b.data(), 
		            qt2.info, qt2.data());
   return qt2;
}

template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qmid, qt3b.info.qmid);
   contract_qt3_qt3_info_lr(qt3a.info, qt3a.data(), qt3b.info, qt3b.data(),
		            qt2.info, qt2.data());
   return qt2;
}

} // ctns

#endif
