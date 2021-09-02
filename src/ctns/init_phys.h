#ifndef INIT_PHYS_H
#define INIT_PHYS_H

#include "init_rbasis.h"
#include "qtensor/qtensor.h"

namespace ctns{

//
// Physical degree of freedoms related:
// stored order = {|0>,|up,dw>,|up>,|dw>}
//

//
// rbasis for type-0 physical site: 
//
template <typename Tm>
inline renorm_basis<Tm> get_rbasis_phys(const int isym){
   renorm_basis<Tm> rbasis;
   // (N)
   if(isym == 0){
      rbasis.resize(2);
      // |00> & |11>
      rbasis[0].sym = qsym(isym,0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].space.push_back(fock::onstate("11"));
      rbasis[0].coeff = linalg::identity_matrix<Tm>(2);
      // a=|01> & b=|10>
      rbasis[1].sym = qsym(isym,1,0);
      rbasis[1].space.push_back(fock::onstate("01")); // a
      rbasis[1].space.push_back(fock::onstate("10")); // b
      rbasis[1].coeff = linalg::identity_matrix<Tm>(2);
   }else if(isym == 1){
      rbasis.resize(3);
      // |00>
      rbasis[0].sym = qsym(isym,0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
      // |11>
      rbasis[1].sym = qsym(isym,2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
      // a=|01> & b=|10>
      rbasis[2].sym = qsym(isym,1,0);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].space.push_back(fock::onstate("10")); // b
      rbasis[2].coeff = linalg::identity_matrix<Tm>(2);
   }else if(isym == 2){
      rbasis.resize(4);
      // |00>
      rbasis[0].sym = qsym(isym,0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
      // |11>
      rbasis[1].sym = qsym(isym,2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
      // |01>
      rbasis[2].sym = qsym(isym,1,1);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].coeff = linalg::identity_matrix<Tm>(1);
      // |10>
      rbasis[3].sym = qsym(isym,1,-1);
      rbasis[3].space.push_back(fock::onstate("10")); // b
      rbasis[3].coeff = linalg::identity_matrix<Tm>(1);
   }
   return rbasis;
}

//
// exact right/left boundary tensor:
//
inline qbond get_qbond_vac(const int isym){ 
   return qbond({{qsym(isym,0,0),1}}); 
}

inline qbond get_qbond_phys(const int isym){
   qbond qphys;
   if(isym == 0){
      qphys.dims = {{qsym(isym,0,0),2},   // 0,2
		    {qsym(isym,1,0),2}};  // a,b
   }else if(isym == 1){
      qphys.dims = {{qsym(isym,0,0),1},   // 0
		    {qsym(isym,2,0),1},   // 2
		    {qsym(isym,1,0),2}};  // a,b
   }else if(isym == 2){
      qphys.dims = {{qsym(isym,0,0),1},   // 0
		    {qsym(isym,2,0),1},   // 2
		    {qsym(isym,1,1),1},   // a
		    {qsym(isym,1,-1),1}}; // b
   }
   return qphys;
}

//
//        n             |vac>
//        |               |
//     ---*---|vac>   n---*
//  |out> 	          |
//
template <typename Tm>
void get_right_bsite(const int isym, qtensor3<Tm>& qt3){
   auto qvac = get_qbond_vac(isym);
   auto qphys = get_qbond_phys(isym);
   qt3.init(qsym(isym), qphys, qphys, qvac);
   for(int bk=0; bk<qphys.size(); bk++){
      int pdim = qphys.get_dim(bk);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(pdim,1);
	 mat(im,0) = 1.0;
         qt3(bk,bk,0)[im] = mat;
      }
   }
}

template <typename Tm>
inline void get_left_bsite(const int isym, qtensor3<Tm>& qt3){
   std::vector<bool> dir = {1,1,0};
   auto qvac = get_qbond_vac(isym);
   auto qphys = get_qbond_phys(isym);
   qt3.init(qsym(isym), qphys, qvac, qphys, dir);
   for(int bk=0; bk<qphys.size(); bk++){
      int pdim = qphys.get_dim(bk);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(1,pdim);
	 mat(0,im) = 1.0;
         qt3(bk,0,bk)[im] = mat;
      }
   }
}

/*
// relations among (idx,occ,mdx), used when physical degree is specified

inline void idx2occ(fock::onstate& state,
		    const int k, 
		    const int idx){
   if(idx == 0){
      state[2*k] = 0; state[2*k+1] = 0; // 0
   }else if(idx == 1){
      state[2*k] = 1; state[2*k+1] = 1; // 2
   }else if(idx == 2){
      state[2*k] = 1; state[2*k+1] = 0; // a
   }else if(idx == 3){
      state[2*k] = 0; state[2*k+1] = 1; // b
   }
}

// block index (bm,im) for middle physical index mdx=(qi,iq)
inline std::pair<int,int> idx2mdx(const int isym, const int idx){
   int qi = (isym==1 and idx==3)? 2 : idx;
   int iq = (isym==1 and idx==3)? 1 : 0;
   return std::make_pair(qi, iq);
}

inline std::pair<int,int> occ2mdx(const int isym,
				  const fock::onstate& state, 
				  const int k){
   std::pair<int,int> mdx;
   if(state[2*k] == 0 and state[2*k+1] == 0){ 	    // 0
      mdx = std::make_pair(0,0);
   }else if(state[2*k] == 1 and state[2*k+1] == 1){ // 2
      mdx = std::make_pair(1,0);
   }else if(state[2*k] == 1 and state[2*k+1] == 0){ // a
      mdx = std::make_pair(2,0);
   }else if(state[2*k] == 0 and state[2*k+1] == 1){ // b
      mdx = (isym == 1)? std::make_pair(2,1) : std::make_pair(3,0);
   }
   return mdx;
}
*/

} // ctns

#endif
