#ifndef CTNS_SDIAG_H
#define CTNS_SDIAG_H

/*
   Algorithms for CTNS:

   4. rcanon_Sdiag_exact:
      rcanon_Sdiag_sample: compute Sdiag via sampling
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"
#include "ctns_expand.h"

namespace ctns{

   // Algorithm 3:
   // exact computation of Sdiag, only for small system
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      double rcanon_Sdiag_exact(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         std::cout << "\nctns::rcanon_Sdiag_exact iroot=" << iroot
            << " thresh_print=" << thresh_print << std::endl;

         // expand CTNS into determinants
         auto expansion = rcanon_expand_onstate(icomb, iroot, thresh_print);
         auto coeff = expansion.second;
         size_t dim = coeff.size();
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

   // compute diagonal entropy via sampling:
   // S = -p[i]log2p[i] = - (sum_i p[i]) <log2p[i] > = -<psi|psi>*<log2p[i]>
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
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
