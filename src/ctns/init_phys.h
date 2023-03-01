#ifndef INIT_PHYS_H
#define INIT_PHYS_H

#include "init_rbasis.h"

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

   //        n             |vac>
   //        |               |
   //     ---*---|vac>   n---*
   //  |out> 	          |
   template <typename Tm>
      stensor3<Tm> get_right_bsite(const int isym){
         auto qvac = get_qbond_vac(isym);
         auto qphys = get_qbond_phys(isym);
         stensor3<Tm> qt3(qsym(isym), qphys, qvac, qphys); // lrc
         for(int bk=0; bk<qphys.size(); bk++){
            auto blk = qt3(bk,0,bk);
            for(int im=0; im<qphys.get_dim(bk); im++){
               blk(im,0,im) = 1.0;
            }
         }
         return qt3;
      }

   //        n            
   //        |            
   //     ---*--- <out|
   //  <vac| 	       
   template <typename Tm>
      stensor3<Tm> get_left_bsite(const int isym){
         auto qvac = get_qbond_vac(isym);
         auto qphys = get_qbond_phys(isym);
         stensor3<Tm> qt3(qsym(isym), qvac, qphys, qphys, dir_LCF);
         for(int bk=0; bk<qphys.size(); bk++){
            auto blk = qt3(0,bk,bk);
            for(int im=0; im<qphys.get_dim(bk); im++){
               blk(0,im,im) = 1.0;
            }
         }
         return qt3;
      }

   //
   // relations among (idx->occ->mdx), used when physical degree is specified in ctns_alg.h
   //
   //  idx = 0,1,2,3
   //  occ = 00,11,01,10
   //  mdx = (qi,iq) [qi-th symmetry block, iq-th state]
   //
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
      std::pair<int,int> mdx;
      if(isym == 0){
         if(idx == 0){
            mdx = std::make_pair(0,0);
         }else if(idx == 1){
            mdx = std::make_pair(0,1);
         }else if(idx == 2){
            mdx = std::make_pair(1,0);
         }else if(idx == 3){
            mdx = std::make_pair(1,1);
         }
      }else if(isym == 1){
         if(idx == 0){
            mdx = std::make_pair(0,0);
         }else if(idx == 1){
            mdx = std::make_pair(1,0);
         }else if(idx == 2){
            mdx = std::make_pair(2,0);
         }else if(idx == 3){
            mdx = std::make_pair(2,1);
         }
      }else if(isym == 2){
         if(idx == 0){
            mdx = std::make_pair(0,0);
         }else if(idx == 1){
            mdx = std::make_pair(1,0);
         }else if(idx == 2){
            mdx = std::make_pair(2,0);
         }else if(idx == 3){
            mdx = std::make_pair(3,0);
         }
      }
      return mdx;
   }

   inline std::pair<int,int> occ2mdx(const int isym,
         const fock::onstate& state, 
         const int k){
      std::pair<int,int> mdx;
      if(state[2*k] == 0 and state[2*k+1] == 0){ 	    // 0
         mdx = idx2mdx(isym, 0);
      }else if(state[2*k] == 1 and state[2*k+1] == 1){ // 2
         mdx = idx2mdx(isym, 1);
      }else if(state[2*k] == 1 and state[2*k+1] == 0){ // a
         mdx = idx2mdx(isym, 2);
      }else if(state[2*k] == 0 and state[2*k+1] == 1){ // b
         mdx = idx2mdx(isym, 3);
      }
      return mdx;
   }

} // ctns

#endif
