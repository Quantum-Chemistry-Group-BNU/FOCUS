#ifndef CTNS_PHYS_H
#define CTNS_PHYS_H

#include "ctns_qsym.h"
#include "qtensor.h"

namespace ctns{

// physical degree of freedoms related

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

/*
template <typename Tm>
qtensor3<Tm> get_left_bsite(){
   std::vector<bool> dir = {1,1,0};
   auto qs_vac = get_qbond_vac();
   auto qs_phys = get_qbond_phys<Tm>();
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


// block index (bm,im) for middle physical index
template <typename Tm>
std::pair<int,int> get_mdx(const int idx){
   std::pair<int,int> mdx;
   if(idx == 0){
      mdx = std::make_pair(0,0);
   }else if(idx == 1){
      mdx = std::make_pair(1,0);
   }else if(idx == 2){
      mdx = std::make_pair(2,0);
   }else if(idx == 3){
      const bool Htype = tools::is_complex<Tm>();
      mdx = Htype? std::make_pair(2,1) : std::make_pair(3,0);
   }
   return mdx;
}

 // 0: |0>=(0,0) -> 0
 // 1: |b>=(0,1) -> 3
 // 2: |a>=(1,0) -> 2
 // 3: |2>=(1,1) -> 1
template <typename Tm>
std::pair<int,int> get_mdx_phys(const fock::onstate& state,
			        const int k){
   int packed = 2*state[2*k]+state[2*k+1];
   const std::vector<int> index = {0,3,2,1};
   return get_mdx<Tm>(index[packed]);
}

// used in random sampling  
inline void assign_occupation_phys(fock::onstate& state,
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
*/

} // ctns

#endif
