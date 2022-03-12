#ifndef CONTRACT_QT3_QT2_H
#define CONTRACT_QT3_QT2_H

namespace ctns{

// --- contract_qt3_qt2 ---
template <typename Tm>
stensor3<Tm> contract_qt3_qt2(const std::string cpos,
		 	      const stensor3<Tm>& qt3a, 
			      const stensor2<Tm>& qt2,
			      const bool ifdagger=false){
   stensor3<Tm> qt3;
   if(cpos == "l"){
      qt3 = contract_qt3_qt2_l(qt3a, qt2, ifdagger);
   }else if(cpos == "r"){
      qt3 = contract_qt3_qt2_r(qt3a, qt2, ifdagger);
   }else if(cpos == "c"){
      qt3 = contract_qt3_qt2_c(qt3a, qt2, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt3_qt2! cpos=" 
	        << cpos << std::endl;
      exit(1);
   }
   return qt3;
}

template <typename Tm>
stensor3<Tm> contract_qt3_qt2_l(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt3a.dir_row() == !dint);
   assert(qt3a.info.qrow == qint);
   qsym sym = qt3a.info.sym + sym2;
   direction3 dir = {dext, qt3a.dir_col(), qt3a.dir_mid()};
   stensor3<Tm> qt3(sym, qext, qt3a.info.qcol, qt3a.info.qmid, dir);
   contract_qt3_qt2_info_l(qt3a.info, qt3a.data(), qt2.info, qt2.data(),
		           qt3.info, qt3.data(), 1.0, false, ifdagger); 
   return qt3;
}

template <typename Tm>
stensor3<Tm> contract_qt3_qt2_r(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt3a.dir_col() == !dint);
   assert(qt3a.info.qcol == qint);
   qsym sym = qt3a.info.sym + sym2;
   direction3 dir = {qt3a.dir_row(), dext, qt3a.dir_mid()};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qext, qt3a.info.qmid, dir);
   contract_qt3_qt2_info_r(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
		           qt3.info, qt3.data(), 1.0, false, ifdagger); 
   return qt3;
}

template <typename Tm>
stensor3<Tm> contract_qt3_qt2_c(const stensor3<Tm>& qt3a, 
			 	const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt3a.dir_mid() == !dint);
   assert(qt3a.info.qmid == qint);
   qsym sym = qt3a.info.sym + sym2;
   direction3 dir = {qt3a.dir_row(), qt3a.dir_col(), dext};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qt3a.info.qcol, qext, dir);
   contract_qt3_qt2_info_c(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
		           qt3.info, qt3.data(), 1.0, false, ifdagger); 
   return qt3;
}

} // ctns

#endif
