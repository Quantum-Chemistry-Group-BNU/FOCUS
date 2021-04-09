#ifndef CTNS_PHYS_H
#define CTNS_PHYS_H

#include "ctns_qsym.h"
#include "ctns_rbasis.h"
#include "qtensor.h"

namespace ctns{

inline qbond get_qbond_vac(){ return qbond({{qsym(0,0),1}}); }

// Physical degree of freedoms related:
// stored order = {|0>,|up,dw>,|up>,|dw>}

inline qbond get_qbond_phys(const int isym){
   qbond qphys;
   if(isym == 1){
      qphys.dims = {{qsym(0,0),1},   // 0
		    {qsym(2,0),1},   // 2
		    {qsym(1,0),2}};  // a,b
   }else if(isym == 2){
      qphys.dims = {{qsym(0,0),1},
		    {qsym(2,0),1},
		    {qsym(1,1),1},
		    {qsym(1,-1),1}};
   }
   return qphys;
}

// rbasis for type-0 physical site: 
template <typename Tm>
inline renorm_basis<Tm> get_rbasis_phys(const int isym){
   renorm_basis<Tm> rbasis;
   // (N)
   if(isym == 1){
      rbasis.resize(3);
      // |00>
      rbasis[0].sym = qsym(0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
      // |11>
      rbasis[1].sym = qsym(2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
      // a=|01> & b=|10>
      rbasis[2].sym = qsym(1,0);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].space.push_back(fock::onstate("10")); // b
      rbasis[2].coeff = linalg::identity_matrix<Tm>(2);
   }else if(isym == 2){
      rbasis.resize(4);
      // |00>
      rbasis[0].sym = qsym(0,0);
      rbasis[0].space.push_back(fock::onstate("00"));
      rbasis[0].coeff = linalg::identity_matrix<Tm>(1);
      // |11>
      rbasis[1].sym = qsym(2,0);
      rbasis[1].space.push_back(fock::onstate("11"));
      rbasis[1].coeff = linalg::identity_matrix<Tm>(1);
      // |01>
      rbasis[2].sym = qsym(1,1);
      rbasis[2].space.push_back(fock::onstate("01")); // a
      rbasis[2].coeff = linalg::identity_matrix<Tm>(1);
      // |10>
      rbasis[3].sym = qsym(1,-1);
      rbasis[3].space.push_back(fock::onstate("10")); // b
      rbasis[3].coeff = linalg::identity_matrix<Tm>(1);
   }
   return rbasis;
}

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

// exact right/left boundary tensor:
//        n             |vac>
//        |               |
//     ---*---|vac>   n---*
//  |out> 	          |
template <typename Tm>
inline void get_right_bsite(const int isym, qtensor3<Tm>& qt3){
   auto qvac = get_qbond_vac();
   auto qphys = get_qbond_phys(isym);
   qt3.init(qsym(), qphys, qphys, qvac);
   for(int k=0; k<qphys.size(); k++){
      int pdim = qphys.get_dim(k);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(pdim,1);
	 mat(im,0) = 1.0;
         qt3(k,k,0)[im] = mat;
      }
   }
}

template <typename Tm>
inline void get_left_bsite(const int isym, qtensor3<Tm>& qt3){
   std::vector<bool> dir = {1,1,0};
   auto qvac = get_qbond_vac();
   auto qphys = get_qbond_phys(isym);
   qt3.init(qsym(), qphys, qvac, qphys, dir);
   for(int k=0; k<qphys.size(); k++){
      int pdim = qphys.get_dim(k);
      for(int im=0; im<pdim; im++){
	 linalg::matrix<Tm> mat(1,pdim);
	 mat(0,im) = 1.0;
         qt3(k,0,k)[im] = mat;
      }
   }
}

} // ctns

#endif
