#ifndef CTNS_SAMPLE_H
#define CTNS_SAMPLE_H

/*
   Algorithms for CTNS:

   3. rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"

namespace ctns{

   // Sampling CTNS to get {|det>,coeff(det)=<det|Psi[i]>} 
   // In case that CTNS is unnormalized, p(det) is also unnormalized. 
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      std::pair<fock::onstate,Tm> rcanon_random(const comb<Qm,Tm>& icomb, 
            const int iroot,
            const bool debug=false){
         if(debug) std::cout << "\nctns::rcanon_random iroot=" << iroot << std::endl; 
         fock::onstate state(2*icomb.get_nphysical());
         // initialize boundary wf for i-th state
         auto wf = icomb.rwfuns[iroot];
         const auto& nodes = icomb.topo.nodes; 
         const auto& rindex = icomb.topo.rindex;
         // loop from left to right
         for(int i=0; i<icomb.topo.nbackbone; i++){
            int tp = nodes[i][0].type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // compute probability for physical index
               std::vector<qtensor2<Qm::ifabelian,Tm>> qt2n(4);
               std::vector<double> weights(4);
               for(int idx=0; idx<4; idx++){
                  qt2n[idx] = qt3.fix_mid( idx2mdx(Qm::isym, idx) );
                  // \sum_a |psi[n,a]|^2
                  auto psi2 = qt2n[idx].dot(qt2n[idx].H()); 
                  weights[idx] = std::real(psi2(0,0)(0,0));
               }
               std::discrete_distribution<> dist(weights.begin(),weights.end());
               int idx = dist(tools::generator);
               idx2occ(state, nodes[i][0].pindex, idx);
               wf = std::move(qt2n[idx]);
            }else if(tp == 3){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qt3 = contract_qt3_qt2("l",site,wf);
               // propogate upwards
               for(int j=1; j<nodes[i].size(); j++){
                  const auto& sitej = icomb.sites[rindex.at(std::make_pair(i,j))];
                  // compute probability for physical index
                  std::vector<qtensor3<Qm::ifabelian,Tm>> qt3n(4);
                  std::vector<double> weights(4);
                  for(int idx=0; idx<4; idx++){
                     auto qt2 = sitej.fix_mid( idx2mdx(Qm::isym, idx) );
                     // purely change direction
                     qt3n[idx] = contract_qt3_qt2("c",qt3,qt2.P()); 
                     // \sum_ab |psi[n,a,b]|^2
                     auto psi2 = contract_qt3_qt3("cr",qt3n[idx],qt3n[idx]); 
                     weights[idx] = std::real(psi2(0,0)(0,0));
                  }
                  std::discrete_distribution<> dist(weights.begin(),weights.end());
                  int idx = dist(tools::generator);
                  idx2occ(state, nodes[i][j].pindex, idx);
                  qt3 = std::move(qt3n[idx]);
               } // j
               wf = qt3.fix_mid(std::make_pair(0,0));
            } // tp
         }
         // finally wf should be the corresponding CI coefficients
         double sgn = state.permute_sgn(icomb.topo.image2); // from orbital ordering
         auto coeff0 = sgn*wf(0,0)(0,0);
         if(debug){
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0=" << coeff0 
               << " coeff1=" << coeff1 
               << " diff=" << coeff0-coeff1 
               << std::endl;
            assert(std::abs(coeff0-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

} // ctns

#endif
