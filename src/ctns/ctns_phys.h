#ifndef CTNS_PHYS_H
#define CTNS_PHYS_H

#include "ctns_qsym.h"
#include "ctns_rbasis.h"
#include "ctns_qtensor.h"

namespace ctns{

// physical degree of freedoms related

// rbasis for type-0 physical site 
template <typename Tm>
renorm_basis<Tm> get_rbasis_phys(){
   const bool Htype = tools::is_complex<Tm>();
   renorm_basis<Tm> rbasis(2);
   rbasis[0].sym = qsym(0,0);
   rbasis[0].space.push_back(fock::onstate("00"));
   rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
   rbasis[1].sym = qsym(2,0);
   rbasis[1].space.push_back(fock::onstate("11"));
   rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
   if(Htype){
      rbasis.resize(3);
      rbasis[2].sym = qsym(1,0);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].space.push_back(fock::onstate("10")); // b
      rbasis[2].coeff = linalg::identity_matrix<Tm>(2);
   }else{
      rbasis.resize(4);
      rbasis[2].sym = qsym(1,1);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].coeff = linalg::identity_matrix<Tm>(1);
      rbasis[3].sym = qsym(1,-1);
      rbasis[3].space.push_back(fock::onstate("10")); // b
      rbasis[3].coeff = linalg::identity_matrix<Tm>(1);
   }
   return rbasis;
}

// qsym_space
qsym_space get_qsym_space_vac(){
   qsym_space qs_vac({{qsym(0,0),1}});
   return qs_vac;
}

template <typename Tm>
qsym_space get_qsym_space_phys(){
   const bool Htype = tools::is_complex<Tm>();
   qsym_space qs_phys;
   if(Htype){
      qs_phys.dims = {{qsym(0,0),1},
		      {qsym(2,0),1},
		      {qsym(1,0),2}};
   }else{
      qs_phys.dims = {{qsym(0,0),1},
		      {qsym(2,0),1},
		      {qsym(1,1),1},
		      {qsym(1,-1),1}};
   }
   return qs_phys;
}

// exact right/left boundary tensor:
//        n             |vac>
//        |               |
//     ---*---|vac>   n---*
//  |out> 	          |
template <typename Tm>
qtensor3<Tm> get_right_bsite(){
   qsym_space qs_vac = get_qsym_space_vac();
   qsym_space qs_phys = get_qsym_space_phys<Tm>();
   qtensor3<Tm> qt3(qsym(0,0), qs_phys, qs_phys, qs_vac);
   for(int k=0; k<qs_phys.size(); k++){
      int pdim = qs_phys.get_dim(k);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(pdim,1);
	 mat(im,0) = 1.0;
         qt3(k,k,0)[im] = mat;
      }
   }
   return qt3;
}

template <typename Tm>
qtensor3<Tm> get_left_bsite(){
   std::vector<bool> dir = {1,1,0};
   qsym_space qs_vac = get_qsym_space_vac();
   qsym_space qs_phys = get_qsym_space_phys<Tm>();
   qtensor3<Tm> qt3(qsym(0,0), qs_phys, qs_vac, qs_phys, dir);
   for(int k=0; k<qs_phys.size(); k++){
      int pdim = qs_phys.get_dim(k);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(1,pdim);
	 mat(0,im) = 1.0;
         qt3(k,0,k)[im] = mat;
      }
   }
   return qt3;
}

} // ctns

#endif
