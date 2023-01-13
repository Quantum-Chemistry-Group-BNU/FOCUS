#ifndef VMC_ELOC_H
#define VMC_ELOC_H

#include "ansatz.h"

namespace vmc{

   template<typename Tm>
      std::vector<std::complex<double>> get_eloc(BaseAnsatz& wavefun,
            const fock::onspace& space,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double& ecore,
            const sci::heatbath_table<Tm>& hbtab,
            const double eps2){
         std::cout << "\nvmc::get_eloc" << std::endl; 

         /*
         fock::onspace space2 = fock::get_fci_space(6,3,3);
         auto H = fock::get_Hmat(space2,int2e,int1e,ecore);
         for(int i=0; i<400; i++){
            std::cout << "i=" << i << " H0i=" << H(20,i) << std::endl;
         }
         */

         // assuming particle number conserving space
         fock::onstate state = space[0];
         int no = state.nelec(), k = state.size(), nv = k - no;
         std::vector<int> olst(no), vlst(nv);
         int nsingles = no*nv;
         int vdim = space.size();
         std::vector<std::complex<double>> eloc(vdim); 
         // loop over <i|H|j> psi(j)/psi(i)
         for(int idx=0; idx<vdim; idx++){
            state = space[idx];
            state.get_olst(olst.data());
            state.get_vlst(vlst.data());
            auto psi_i = wavefun.psi(state);
            double v0i = std::abs(psi_i);
            eloc[idx] = ecore + fock::get_Hii(state,int2e,int1e);
            // singles
            for(int ia=0; ia<nsingles; ia++){
               int ix = ia%no, ax = ia/no;
               int i = olst[ix], a = vlst[ax];
               fock::onstate state1(state);
               state1[i] = 0;
               state1[a] = 1;
               auto pr = fock::get_HijS(state,state1,int2e,int1e);
               eloc[idx] += pr.first * wavefun.psi(state1)/psi_i;
               /*
               std::cout << "ia=" << i << "," << a 
                         << " state1=" << state1.to_string()
                         << " HijS=" << pr.first << std::endl; 
               */
            } // ia 
            // doubles
            for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
               auto pr = tools::inverse_pair0(ijdx);
               int i = olst[pr.first], j = olst[pr.second];
               int ij = tools::canonical_pair0(i,j);
               for(const auto& p : hbtab.eri4.at(ij)){
                  if(p.first*v0i < eps2) break; // avoid searching all doubles
                  auto ab = tools::inverse_pair0(p.second);
                  int a = ab.first, b = ab.second;
                  if(state[a]==0 && state[b]==0){ // if true double excitations
                     fock::onstate state2(state);
                     state2[i] = 0;
                     state2[j] = 0;
                     state2[a] = 1;
                     state2[b] = 1;
                     auto pr = fock::get_HijD(state,state2,int2e,int1e);
                     eloc[idx] += pr.first * wavefun.psi(state2)/psi_i;
                     /*
                     std::cout << "iiab=" << i << "," << j << "," << a << "," << b 
                               << " state2=" << state2.to_string()
                               << " HijD=" << pr.first << std::endl;
                     */
                  }
               } // ab
            } // ij
            /*
            std::cout << "idx=" << idx
                      << " state=" << state
                      << " eloc=" << eloc[idx] 
                      << std::endl;
            */
         } // idx
         return eloc;
      }

} // vmc

#endif
