#ifndef CONTRACT_OPXWF_H
#define CONTRACT_OPXWF_H

namespace ctns{

//
// 
//
template <typename Tm>
stensor3<Tm> contract_opxwf(const std::string cpos,
		 	    const stensor3<Tm>& qt3a, 
			    stensor2<Tm>& qt2b, // for opxwf, qt2b should be changable!
			    const bool iftrans){
   return contract_qt3_qt2(cpos, qt3a, qt2b, iftrans);
}
template <typename Tm>
stensor4<Tm> contract_opxwf(const std::string cpos,
		 	    const stensor4<Tm>& qt4a, 
			    stensor2<Tm>& qt2b,
			    const bool iftrans){
   return contract_qt4_qt2(cpos, qt4a, qt2b, iftrans);
}

template <typename Tm>
void contract_opxwf_info(const std::string cpos,
			 const qinfo2<Tm>& qt2b_info,
			 Tm* qt2b_data,
			 const qinfo3<Tm>& qt3a_info,
			 Tm* qt3a_data,
			 qinfo3<Tm>& qt3_info,
			 Tm* qt3_data,
		         const double alpha,
			 const bool accum,
		         const bool iftrans){
   if(cpos == "l"){
      contract_qt3_qt2_info_l(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, iftrans);
   }else if(cpos == "r"){
      contract_qt3_qt2_info_r(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, iftrans);
   }else if(cpos == "c"){
      contract_qt3_qt2_info_c(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, iftrans);
   }else{
      std::cout << "error: no such case in contract_opxwf_info! cpos=" 
	        << cpos << std::endl;
      exit(1);
   }
}
template <typename Tm>
void contract_opxwf_info(const std::string cpos,
			 const qinfo2<Tm>& qt2b_info,
			 Tm* qt2b_data,
			 const qinfo4<Tm>& qt4a_info,
			 Tm* qt4a_data,
			 qinfo4<Tm>& qt4_info,
			 Tm* qt4_data,
		         const double alpha,
			 const bool accum,
			 const bool iftrans){
   if(cpos == "l"){
      contract_qt4_qt2_info_l(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	      qt4_info, qt4_data, alpha, accum, iftrans);
   }else if(cpos == "r"){
      contract_qt4_qt2_info_r(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	      qt4_info, qt4_data, alpha, accum, iftrans);
   }else if(cpos == "c1"){
      contract_qt4_qt2_info_c1(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	       qt4_info, qt4_data, alpha, accum, iftrans);
   }else if(cpos == "c2"){
      contract_qt4_qt2_info_c2(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	       qt4_info, qt4_data, alpha, accum, iftrans);
   }else{
      std::cout << "error: no such case in contract_opxwf_info! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
}

} // ctns

#endif
