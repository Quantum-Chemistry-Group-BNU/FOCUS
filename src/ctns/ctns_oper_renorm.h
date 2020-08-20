#ifndef CTNS_OPER_RENORM_H
#define CTNS_OPER_RENORM_H

#include "ctns_comb.h"

namespace ctns{

template <typename Tm>
linalg::matrix<Tm> get_Hmat(const comb<Tm>& icomb, 
		            const integral::two_body<Tm>& int2e,
		            const integral::one_body<Tm>& int1e,
		            const double ecore,
		            const std::string scratch){
   std::cout << "\nctns::get_Hmat" << std::endl;
/*
   // environement
   oper_env_right(icomb, int2e, int1e, scratch);
   // load
   oper_dict qops;
   auto p = make_pair(0,0); 
   string fname = oper_fname(scratch, p, "rop");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   Hmat += ecore*linalg::identity_matrix(Hmat.rows());
   return Hmat;
*/
   linalg::matrix<Tm> mat;
   return mat;
}

} // ctns

#endif
