#ifndef CTNS_SDIAG_SU2_H
#define CTNS_SDIAG_SU2_H

/*
   Algorithms for CTNS:

   4. rcanon_Sdiag_exact:
      rcanon_Sdiag_sample: compute Sdiag via sampling
*/

#include "../../core/onspace.h"
#include "../../core/analysis.h"
#include "../ctns_comb.h"
#include "../ctns_expand.h"
#include "ctns_expand_su2.h"
#include "ctns_random_su2.h"

namespace ctns{

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
         std::cout << "dim=" << dim << std::scientific << std::setprecision(3) 
            << " ovlp=" << ovlp 
            << " Sdiag(exact)=" << Sdiag 
            << " IPR=" << IPR 
            << std::endl;
         return Sdiag;
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]logp[i] = - (sum_i p[i]) <logp[i] > = -<psi|psi>*<logp[i]>
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      double rcanon_sample_samps2det(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int nsample=10000,
            const double pthrd=1.e-2,
            const int nprt=10){ // no. of largest states to be printed
         auto t0 = tools::get_time();
         const double cutoff = 0.0;
         std::cout << "\nctns::rcanon_sample_samps2det:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " nsample=" << nsample 
            << " pthrd=" << pthrd
            << " nprt=" << nprt
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
               std::cout << " i=" << i << " TIMING=" << dt << " S" << std::endl;	      
               t0 = tools::get_time();
            }
         }
         // print important determinants
         int size = pop.size();
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
         std::cout << "sampled unique det: pop.size=" << size 
            << " Sdiag[pop]=" << Sdpop << " IPR[pop]=" << IPRpop 
            << std::endl;
         std::cout << "warning: it is not the exact pop (interference is neglected)!" << std::endl;
         // compare the first n important dets by counts
         auto indx = tools::sort_index(counts,1);
         int sum = 0;
         for(int i=0; i<size; i++){
            int idx = indx[i];
            auto state = states[idx];
            double pop = counts[idx]/(1.0*nsample);
            if(pop < pthrd or i >= nprt) break;
            sum += counts[idx];
            std::cout << " i=" << i << " state=" << state
               << " counts=" << counts[idx] 
               << " p_i(sample)=" << pop
               << std::endl;
         }
         std::cout << "accumulated counts for listed confs=" << sum 
            << " nsample=" << nsample 
            << " per=" << 1.0*sum/nsample << std::endl;
         return Sdpop;
      }

} // ctns

#endif
