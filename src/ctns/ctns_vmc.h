#ifndef CTNS_VMC_H
#define CTNS_VMC_H

#include "ci/sci_util.h"

namespace ctns{

   // main for sweep optimizations for CTNS
   template <typename Km>
      void vmc_estimate(comb<Km>& icomb, // initial comb wavefunction
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch){
         using Tm = typename Km::dtype;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         int nsample = schd.ctns.nsample;
         int nroots = icomb.get_nroots(); 
         const bool debug = (rank==0); 
         if(debug){
            std::cout << "\nctns::vmc_estimate nsample=" << nsample
               << " nroots=" << nroots
               << std::endl;
         }
         if(schd.ctns.maxsweep == 0) return;
         auto t0 = tools::get_time();

         // set up head-bath table
         const double eps2 = 1.e-10;
         sci::heatbath_table<Tm> hbtab(int2e, int1e);

         int noff = nsample/10;
         int k = icomb.get_nphysical()*2;
         int no = icomb.get_sym_state().ne();
         int nv = k - no;
         int nsingles = no*nv;
         std::vector<int> olst(no), vlst(nv);
         for(int iroot=0; iroot<nroots; iroot++){
            std::cout << "\niroot=" << iroot << std::endl;

            // generate samples 
            double ene = 0.0, ene2 = 0.0, std = 0.0;
            for(int i=0; i<nsample; i++){
               auto pr = rcanon_random(icomb,iroot);
               auto state = pr.first;
               Tm psi_i = pr.second;
               // given state |i>, loop over <i|H|j> psi(j)/psi(i)
               state.get_olst(olst.data());
               state.get_vlst(vlst.data());
               double v0i = std::abs(psi_i);
               Tm eloc = ecore + fock::get_Hii(state,int2e,int1e);
               // singles
               for(int ia=0; ia<nsingles; ia++){
                  int ix = ia%no, ax = ia/no;
                  int i = olst[ix], a = vlst[ax];
                  fock::onstate state1(state);
                  state1[i] = 0;
                  state1[a] = 1;
                  auto pr = fock::get_HijS(state,state1,int2e,int1e);
                  Tm psi_j = ctns::rcanon_CIcoeff(icomb, state1)[iroot];
                  eloc += pr.first * psi_j/psi_i;
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
                        Tm psi_j = ctns::rcanon_CIcoeff(icomb, state2)[iroot];
                        eloc += pr.first * psi_j/psi_i;
                     }
                  } // ab
               } // ij
               double fac = 1.0/(i+1.0);
               ene = (ene*i + std::real(eloc))*fac;
               ene2 = (ene2*i + std::norm(eloc))*fac; 
               if((i+1)%noff == 0){
                  // Note: <psi|H-E|P><P|H-E|psi> is not <psi|(H-E)^2|psi>,
                  // which can be simply seen by taking |psi> as a single determinant!
                  // Thus, it is not the variance of the wavefunction.
                  std = std::sqrt(std::abs(ene2-ene*ene)/(i+1.e-10));
                  std::cout << " i=" << i 
                     << " <H>=" << std::defaultfloat << std::setprecision(12) << ene 
                     //<< " <(H-E)^2>=" << std::scientific << std::setprecision(3) << (ene2-ene*ene)
                     << " std=" << std::scientific << std::setprecision(3) << std
                     << " range=(" << std::defaultfloat << std::setprecision(12) 
                     << ene-std << "," << ene+std << ")" 
                     << std::endl;
               }
            } // sample

         } // iroot

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::vmc_estimate", t0, t1);
         }
      }

} // ctns

#endif
