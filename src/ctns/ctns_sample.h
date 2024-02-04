#ifndef CTNS_SAMPLE_H
#define CTNS_SAMPLE_H

/*
   Algorithms for CTNS:

   3. rcanon_Sdiag_exact:
      rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
      rcanon_Sdiag_sample: compute Sdiag via sampling
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"

namespace ctns{

   // Algorithm 3:
   // exact computation of Sdiag, only for small system
   template <typename Qm, typename Tm>
      double rcanon_Sdiag_exact(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         std::cout << "\nctns::rcanon_Sdiag_exact iroot=" << iroot
            << " thresh_print=" << thresh_print << std::endl;
         // setup FCI space
         qsym sym_state = icomb.get_sym_state();
         int ne = sym_state.ne(); 
         int ks = icomb.get_nphysical();
         fock::onspace fci_space;
         if(Qm::isym == 0){
            fci_space = fock::get_fci_space(2*ks);
         }else if(Qm::isym == 1){
            fci_space = fock::get_fci_space(2*ks,ne);
         }else if(Qm::isym == 2){
            int tm = sym_state.tm(); 
            int na = (ne+tm)/2, nb = ne-na;
            fci_space = fock::get_fci_space(ks,na,nb); 
         }
         size_t dim = fci_space.size();
         std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI=" << dim << std::endl;

         // brute-force computation of exact coefficients <n|CTNS>
         std::vector<Tm> coeff(dim,0.0);
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
         }
         double Sdiag = fock::coeff_entropy(coeff);
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2); 
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;

         // check: computation by sampling CI vector
         std::vector<double> weights(dim,0.0);
         std::transform(coeff.begin(), coeff.end(), weights.begin(),
               [](const Tm& x){ return std::norm(x); });
         std::discrete_distribution<> dist(weights.begin(),weights.end());
         const int nsample = 1e6;
         int noff = nsample/10;
         const double cutoff = 1.e-12;
         double Sd = 0.0, Sd2 = 0.0, std = 0.0;
         for(int i=0; i<nsample; i++){
            int idx = dist(tools::generator);
            auto ci2 = weights[idx];
            double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
            double fac = 1.0/(i+1.0);
            Sd = (Sd*i + s)*fac;
            Sd2 = (Sd2*i + s*s)*fac;
            if((i+1)%noff == 0){
               std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
               std::cout << " i=" << i << " Sd=" << Sd << " std=" << std 
                  << " range=(" << Sd-std << "," << Sd+std << ")"
                  << std::endl;
            }
         }
         return Sdiag;
      }

   // Sampling CTNS to get {|det>,coeff(det)=<det|Psi[i]>} 
   // In case that CTNS is unnormalized, p(det) is also unnormalized. 
   template <typename Qm, typename Tm>
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
            }
         }
         // finally wf should be the corresponding CI coefficients: coeff0*sgn = coeff1
         auto coeff0 = wf(0,0)(0,0);
         if(debug){
            double sgn = state.permute_sgn(icomb.topo.image2); // from orbital ordering
            auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
            std::cout << " state=" << state 
               << " coeff0,sgn=" << coeff0 << "," << sgn
               << " coeff1=" << coeff1 
               << " diff=" << coeff0*sgn-coeff1 << std::endl;
            assert(std::abs(coeff0*sgn-coeff1)<1.e-10);
         }
         return std::make_pair(state,coeff0);
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]log2p[i] = - (sum_i p[i]) <log2p[i] > = -<psi|psi>*<log2p[i]>
   template <typename Qm, typename Tm>
      double rcanon_Sdiag_sample(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int nsample,  
            const int nprt=10){ // no. of largest states to be printed
         const double cutoff = 1.e-12;
         std::cout << "\nctns::rcanon_Sdiag_sample iroot=" << iroot 
            << " nsample=" << nsample 
            << " nprt=" << nprt << std::endl;
         auto t0 = tools::get_time();
         const int noff = nsample/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
         std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
         // start sampling
         double Sd = 0.0, Sd2 = 0.0, std = 0.0;
         std::map<fock::onstate,int> pop;
         for(int i=0; i<nsample; i++){
            auto pr = rcanon_random(icomb,iroot);
            auto state = pr.first;
            auto ci2 = std::norm(pr.second);
            // statistical analysis
            pop[state] += 1;
            double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
            double fac = 1.0/(i+1.0);
            Sd = (Sd*i + s)*fac;
            Sd2 = (Sd2*i + s*s)*fac;
            if((i+1)%noff == 0){
               std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
               auto t1 = tools::get_time();
               double dt = tools::get_duration(t1-t0);
               std::cout << " i=" << i << " Sd=" << Sd << " std=" << std
                  << " timing=" << dt << " s" << std::endl;	      
               t0 = tools::get_time();
            }
         }
         // print important determinants
         if(nprt > 0){
            int size = pop.size();
            std::cout << "sampled important determinants: pop.size=" << size << std::endl; 
            std::vector<fock::onstate> states(size);
            std::vector<int> counts(size);
            int i = 0;
            for(const auto& pr : pop){
               states[i] = pr.first;
               counts[i] = pr.second;
               i++;
            }
            auto indx = tools::sort_index(counts,1);
            // compare the first n important dets by counts
            int sum = 0;
            for(int i=0; i<std::min(size,nprt); i++){
               int idx = indx[i];
               fock::onstate state = states[idx];
               auto ci = rcanon_CIcoeff(icomb, state)[iroot];
               sum += counts[idx];
               std::cout << " i=" << i << " " << state
                  << " counts=" << counts[idx] 
                  << " p_i(sample)=" << counts[idx]/(1.0*nsample)
                  << " p_i(exact)=" << std::norm(ci)/ovlp 
                  << " c_i(exact)=" << ci/std::sqrt(ovlp)
                  << std::endl;
            }
            std::cout << "accumulated counts=" << sum 
               << " nsample=" << nsample 
               << " per=" << 1.0*sum/nsample << std::endl;
         }
         return Sd;
      }

} // ctns

#endif
