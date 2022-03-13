#ifndef CONTRACT_QT4_QT2_H
#define CONTRACT_QT4_QT2_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2,
			      const bool ifdagger=false){
   stensor4<Tm> qt4;
   if(cpos == "l"){
      qt4 = contract_qt4_qt2_l(qt4a, qt2, ifdagger);
   }else if(cpos == "r"){
      qt4 = contract_qt4_qt2_r(qt4a, qt2, ifdagger);
   }else if(cpos == "c1"){
      qt4 = contract_qt4_qt2_c1(qt4a, qt2, ifdagger);
   }else if(cpos == "c2"){
      qt4 = contract_qt4_qt2_c2(qt4a, qt2, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt4_qt2! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_l(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_row() == !dint);
   assert(qt4a.info.qrow == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qext, qt4a.info.qcol, qt4a.info.qmid, qt4a.info.qver);
   contract_qt4_qt2_info_l(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		           qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_r(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_col() == !dint);
   assert(qt4a.info.qcol == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qext, qt4a.info.qmid, qt4a.info.qver);
   contract_qt4_qt2_info_r(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		           qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c1(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_mid() == !dint);
   assert(qt4a.info.qmid == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qext, qt4a.info.qver);
   contract_qt4_qt2_info_c1(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		            qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c2(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_ver() == !dint);
   assert(qt4a.info.qver == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qt4a.info.qmid, qext);
   contract_qt4_qt2_info_c2(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		            qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

} // ctns

#endif
