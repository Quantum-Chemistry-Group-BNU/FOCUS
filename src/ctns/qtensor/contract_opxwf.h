#ifndef CONTRACT_OPXWF_H
#define CONTRACT_OPXWF_H

namespace ctns{

template <typename Tm>
std::string get_name(){ return ""; }
template <> inline std::string get_name<stensor2<double>>(){ return "st2<real>"; }
template <> inline std::string get_name<stensor3<double>>(){ return "st3<real>"; }
template <> inline std::string get_name<stensor4<double>>(){ return "st4<real>"; }
template <> inline std::string get_name<stensor2<std::complex<double>>>(){ return "st2<cmplx>"; }
template <> inline std::string get_name<stensor3<std::complex<double>>>(){ return "st3<cmplx>"; }
template <> inline std::string get_name<stensor4<std::complex<double>>>(){ return "st4<cmplx>"; }

template <typename Tm>
stensor3<Tm> contract_opxwf(const std::string cpos,
			    const stensor2<Tm>& qt2b,
		 	    const stensor3<Tm>& qt3a, 
			    const bool ifdagger=false){
   stensor3<Tm> qt3;
   if(cpos == "l"){
      qt3 = contract_qt3_qt2_l(qt3a, qt2b, ifdagger);
   }else if(cpos == "r"){
      qt3 = contract_qt3_qt2_r(qt3a, qt2b, ifdagger);
   }else if(cpos == "c"){
      qt3 = contract_qt3_qt2_c(qt3a, qt2b, ifdagger);
   }else{
      std::cout << "error: no such case in contract_opxwf! cpos=" 
	        << cpos << std::endl;
      exit(1);
   }
   return qt3;
}
template <typename Tm>
stensor4<Tm> contract_opxwf(const std::string cpos,
			    const stensor2<Tm>& qt2b,
		 	    const stensor4<Tm>& qt4a, 
			    const bool ifdagger=false){
   stensor4<Tm> qt4;
   if(cpos == "l"){
      qt4 = contract_qt4_qt2_l(qt4a, qt2b, ifdagger);
   }else if(cpos == "r"){
      qt4 = contract_qt4_qt2_r(qt4a, qt2b, ifdagger);
   }else if(cpos == "c1"){
      qt4 = contract_qt4_qt2_c1(qt4a, qt2b, ifdagger);
   }else if(cpos == "c2"){
      qt4 = contract_qt4_qt2_c2(qt4a, qt2b, ifdagger);
   }else{
      std::cout << "error: no such case in contract_opxwf! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
   return qt4;
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
		         const bool ifdagger=false){
   if(cpos == "l"){
      contract_qt3_qt2_info_l(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, ifdagger);
   }else if(cpos == "r"){
      contract_qt3_qt2_info_r(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, ifdagger);
   }else if(cpos == "c"){
      contract_qt3_qt2_info_c(qt3a_info, qt3a_data, qt2b_info, qt2b_data,
		      	      qt3_info, qt3_data, alpha, accum, ifdagger);
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
			 const bool ifdagger=false){
   if(cpos == "l"){
      contract_qt4_qt2_info_l(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	      qt4_info, qt4_data, alpha, accum, ifdagger);
   }else if(cpos == "r"){
      contract_qt4_qt2_info_r(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	      qt4_info, qt4_data, alpha, accum, ifdagger);
   }else if(cpos == "c1"){
      contract_qt4_qt2_info_c1(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	       qt4_info, qt4_data, alpha, accum, ifdagger);
   }else if(cpos == "c2"){
      contract_qt4_qt2_info_c2(qt4a_info, qt4a_data, qt2b_info, qt2b_data,
		      	       qt4_info, qt4_data, alpha, accum, ifdagger);
   }else{
      std::cout << "error: no such case in contract_opxwf_info! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
}

} // ctns

#endif
