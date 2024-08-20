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
#include "ctns_random.h"

namespace ctns{

   // --- Sdiag_exact: Abelian MPS ---

   // Algorithm 3:
   // exact computation of Sdiag, only for small system
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      double rcanon_Sdiag_exact(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double pthrd=1.e-2){
         std::cout << "\nctns::rcanon_Sdiag_exact:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot
            << " pthrd=" << pthrd
            << std::endl;
         // expand CTNS into determinants
         auto expansion = rcanon_expand_onspace(icomb, iroot, pthrd);
         auto coeffs = expansion.second;
         size_t dim = coeffs.size();
         double Sdiag = fock::coeff_entropy(coeffs);
         double ovlp = std::pow(linalg::xnrm2(dim,&coeffs[0]),2);
         std::transform(coeffs.begin(), coeffs.end(), coeffs.begin(),
              [](const Tm& x){ return std::norm(x); });
         double IPR = std::pow(linalg::xnrm2(dim,&coeffs[0]),2); 
         std::cout << "dim=" << dim << std::setprecision(6) 
            << " ovlp=" << ovlp 
            << " Sdiag(exact)=" << Sdiag
            << " IPR=" << IPR 
            << std::endl;
         return Sdiag;
      }

   // --- Sdiag_exact: Non-Abelian MPS --- 
   
   // exact computation of Sdiag, only for small system
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      double rcanon_Sdiag_exact(const comb<Qm,Tm>& icomb,
            const int iroot,
            const std::string type,
            const double pthrd=1.e-2){
         std::cout << "\nctns::rcanon_Sdiag_exact(su2):"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot
            << " type=" << type
            << " pthrd=" << pthrd 
            << std::endl;
         // expand CTNS into csf/det
         std::vector<Tm> coeffs;
         if(type == "csf"){
            auto expansion = rcanon_expand_csfspace(icomb, iroot, pthrd);
            coeffs = expansion.second;
         }else if(type == "det"){
            auto expansion = rcanon_expand_onspace(icomb, iroot, pthrd);
            coeffs = expansion.second;
         }else{
            tools::exit("error: no such type for rcanon_Sdiag_exact");
         } 
         size_t dim = coeffs.size();
         double Sdiag = fock::coeff_entropy(coeffs);
         double ovlp = std::pow(linalg::xnrm2(dim,&coeffs[0]),2);
         std::transform(coeffs.begin(), coeffs.end(), coeffs.begin(),
              [](const Tm& x){ return std::norm(x); });
         double IPR = std::pow(linalg::xnrm2(dim,&coeffs[0]),2); 
         std::cout << "dim=" << dim << std::setprecision(6) 
            << " ovlp=" << ovlp 
            << " Sdiag(exact)=" << Sdiag 
            << " IPR=" << IPR 
            << std::endl;
         return Sdiag;
      }


   // --- Sdiag_sample: Abelian & Non-Abelian MPS ---

   // compute diagonal entropy via sampling:
   // S = -p[i]logp[i] = - (sum_i p[i]) <logp[i] > = -<psi|psi>*<logp[i]>
   template <typename Qm, typename Tm>
      double rcanon_Sdiag_sample(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int nsample=10000,
            const double pthrd=1.e-2){ // no. of largest states to be printed
         using statetype = typename std::conditional<Qm::ifabelian, fock::onstate, fock::csfstate>::type; 
         auto t0 = tools::get_time();
         const double cutoff = 0.0;
         std::cout << "\nctns::rcanon_Sdiag_sample:" 
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " nsample=" << nsample 
            << " pthrd=" << pthrd
            << std::endl;
         const int noff = nsample/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
         std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
         // start sampling
         double Sd = 0.0, Sd2 = 0.0, std = 0.0, IPR = 0.0;
         std::map<statetype,int> pop;
         for(int i=0; i<nsample; i++){
            auto pr = rcanon_random(icomb,iroot);
            auto state = pr.first;
            //std::cout << "i=" << i << " state=" << state << " cicoeff=" << pr.second << std::endl;
            auto ci2 = std::norm(pr.second);
            // statistical analysis
            pop[state] += 1;
            double s = (ci2 < cutoff)? 0.0 : -log(ci2)*ovlp;
            double ipr = ci2*ovlp;
            double fac = 1.0/(i+1.0);
            Sd = (Sd*i + s)*fac;
            Sd2 = (Sd2*i + s*s)*fac;
            IPR = (IPR*i + ipr)*fac;
            if((i+1)%noff == 0){
               std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
               auto t1 = tools::get_time();
               double dt = tools::get_duration(t1-t0);
               std::cout << " i=" << i 
                  << std::setprecision(6)
                  << " Sdiag=" << Sd << " std=" << std
                  << " IPR=" << IPR
                  << " timing=" << dt << " s" << std::endl;	      
               t0 = tools::get_time();
            }
         }
         // print important determinants
         int size = pop.size();
         std::cout << "sampled important csf/det: pop.size=" << size << std::endl; 
         std::vector<statetype> states(size);
         std::vector<int> counts(size);
         double Sdpop = 0.0, IPRpop = 0.0;
         int i = 0;
         for(const auto& pr : pop){
            states[i] = pr.first;
            counts[i] = pr.second;
            double ci2 = counts[i]/(1.0*nsample); // frequency in the sample
            Sdpop += (ci2 < cutoff)? 0.0 : -ci2*log(ci2)*ovlp;
            IPRpop += ci2*ci2*ovlp;
            i++;
         }
         auto indx = tools::sort_index(counts,1);
         // compare the first n important dets by counts
         int sum = 0;
         for(int i=0; i<size; i++){
            int idx = indx[i];
            auto state = states[idx];
            //std::cout << "i=" << i << " state=" << state << " iroot=" << iroot << std::endl;
            auto ci = rcanon_CIcoeff(icomb, state)[iroot];
            //std::cout << "ci=" << ci << std::endl;
            double pop = std::norm(ci)/ovlp;
            if(pop < pthrd) break;
            sum += counts[idx];
            std::cout << " i=" << i << " " << state
               << " counts=" << counts[idx] 
               << " p_i(sample)=" << counts[idx]/(1.0*nsample)
               << " p_i(exact)=" << pop
               << " c_i(exact)=" << ci/std::sqrt(ovlp)
               << std::endl;
         }
         std::cout << "accumulated counts=" << sum 
            << " nsample=" << nsample 
            << " per=" << 1.0*sum/nsample << std::endl;
         std::cout << "estimated Sdiag[MC]=" << Sd << " Sdiag[pop]=" << Sdpop << std::endl;
         std::cout << "estimated IPR[MC]=" << IPR << " IPR[pop]=" << IPRpop << std::endl;
         return Sd;
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]logp[i] = - (sum_i p[i]) <logp[i] > = -<psi|psi>*<logp[i]>
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      double rcanon_sample_samps2det(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int nsample=10000,
            const double pthrd=1.e-2){ // no. of largest states to be printed
         auto t0 = tools::get_time();
         const double cutoff = 0.0;
         std::cout << "\nctns::rcanon_sample_samps2det:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " nsample=" << nsample 
            << " pthrd=" << pthrd
            << std::endl;
         const int noff = nsample/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
         std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
         // start sampling
         std::map<fock::onstate,int> pop;
         for(int i=0; i<nsample; i++){
            // two-step sampling 
            auto pr = rcanon_random(icomb,iroot);
            auto csf = pr.first;
            auto pr2 = csf.random();
            auto state = pr2.first;
            // statistical analysis
            pop[state] += 1;
            if((i+1)%noff == 0){
               auto t1 = tools::get_time();
               double dt = tools::get_duration(t1-t0);
               std::cout << " i=" << i << " timing=" << dt << " s" << std::endl;	      
               t0 = tools::get_time();
            }
         }
         // print important determinants
         int size = pop.size();
         std::cout << "sampled important det: pop.size=" << size << std::endl; 
         std::vector<fock::onstate> states(size);
         std::vector<int> counts(size);
         double Sdpop = 0.0, IPRpop = 0.0;
         int i = 0;
         for(const auto& pr : pop){
            states[i] = pr.first;
            counts[i] = pr.second;
            double ci2 = counts[i]/(1.0*nsample);
            Sdpop += (ci2 < cutoff)? 0.0 : -ci2*log(ci2)*ovlp;
            IPRpop += ci2*ci2*ovlp;
            i++;
         }
         auto indx = tools::sort_index(counts,1);
         // compare the first n important dets by counts
         int sum = 0;
         for(int i=0; i<size; i++){
            int idx = indx[i];
            auto state = states[idx];
            double pop = counts[idx]/(1.0*nsample);
            if(pop < pthrd) break;
            sum += counts[idx];
            std::cout << " i=" << i << " " << state
               << " counts=" << counts[idx] 
               << " p_i(sample)=" << pop
               << std::endl;
         }
         std::cout << "accumulated counts=" << sum 
            << " nsample=" << nsample 
            << " per=" << 1.0*sum/nsample << std::endl;
         std::cout << "estimated Sdiag[pop]=" << Sdpop << std::endl;
         std::cout << "warning: it is not the exact pop (interference is neglected)!" << std::endl;
         std::cout << "estimated IPR[pop]=" << IPRpop << std::endl;
         return Sdpop;
      }

} // ctns

#endif
