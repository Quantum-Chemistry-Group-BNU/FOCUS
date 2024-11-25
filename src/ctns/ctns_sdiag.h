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
         std::cout << "dim=" << dim << std::scientific << std::setprecision(3) 
            << " ovlp=" << ovlp 
            << " Sdiag(exact)=" << Sdiag
            << " IPR=" << IPR 
            << std::endl;
         return Sdiag;
      }

   // --- Sdiag_sample: Abelian & Non-Abelian MPS ---

   template <typename statetype, typename Tm>
      void print_samples(const std::string statestr,
            const double ovlp,
            std::map<statetype,int>& pop,
            std::map<statetype,Tm>& coeff,
            const int nsample,
            const double pthrd,
            const int nprt,
            const std::string saveconfs){
         const double cutoff = 0.0;
         // convert map to vector 
         size_t size = pop.size();
         std::vector<statetype> states(size);
         std::vector<int> counts(size);
         std::vector<double> coeff2(size); 
         double Sdpop = 0.0, IPRpop = 0.0;
         size_t i = 0;
         for(const auto& pr : pop){
            states[i] = pr.first;
            counts[i] = pr.second;
            coeff2[i] = std::norm(coeff[pr.first]);
            double ci2 = counts[i]/(1.0*nsample); // frequency in the sample
            Sdpop += (ci2 < cutoff)? 0.0 : -ci2*log(ci2)*ovlp;
            IPRpop += ci2*ci2*ovlp;
            i++;
         }
         double psum = std::accumulate(coeff2.begin(), coeff2.end(), 0.0);
         std::cout << "sampled unique " << statestr << ": pop.size=" << size 
            << " psum=" << psum << " Sdiag[pop]=" << Sdpop << " IPR[pop]=" << IPRpop 
            << std::endl;
         // print the first n important configurations
         auto indx = tools::sort_index(coeff2,1);
         int sum = 0;
         for(int i=0; i<size; i++){
            int idx = indx[i];
            auto state = states[idx];
            auto ci = coeff[state];
            double pi = coeff2[idx]/ovlp;
            if(pi < pthrd or i >= nprt) break;
            sum += counts[idx];
            std::cout << " i=" << i << " state=" << state
               << " c_i(exact)=" << std::scientific << std::setw(10) 
               << std::setprecision(3) << ci/std::sqrt(ovlp)
               << " p_i(exact)=" << pi
               << " p_i(sample)=" << counts[idx]/(1.0*nsample)
               << " counts=" << counts[idx] 
               << std::endl;
         }
         std::cout << "accumulated counts for listed confs=" << sum 
            << " nsample=" << nsample 
            << " per=" << 1.0*sum/nsample << std::endl;
         // save configurations to text file
         if(!saveconfs.empty()){
            std::cout << "save to file " << saveconfs << ".txt" << std::endl;
            std::ofstream file(saveconfs+".txt");
            file << std::scientific << std::setprecision(12);
            file << "size= " << size << " psum= " << psum << std::endl;
            for(int i=0; i<size; i++){
               int idx = indx[i];
               const auto& state = states[idx];
               const auto& ci = coeff[state];
               file << state << " " << ci << std::endl;
            }
            file.close(); 
         }
      }

   // compute diagonal entropy via sampling:
   // S = -p[i]logp[i] = - (sum_i p[i]) <logp[i] > = -<psi|psi>*<logp[i]>
   template <typename Qm, typename Tm>
      double rcanon_Sdiag_sample(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int nsample=10000,
            const double pthrd=1.e-2,
            const int nprt=10, // no. of largest states to be printed
            const std::string saveconfs=""){
         auto ti = tools::get_time();
         std::string statestr = Qm::ifabelian? "det" : "csf";
         using statetype = typename std::conditional<Qm::ifabelian, fock::onstate, fock::csfstate>::type; 
         std::cout << "\nctns::rcanon_Sdiag_sample:" 
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " nsample=" << nsample 
            << " pthrd=" << pthrd
            << " nprt=" << nprt
            << " saveconfs=" << saveconfs
            << std::endl;
         
         // check
         if(iroot > icomb.get_nroots()-1){
            std::cout << "eroor: iroot exceeds nroots=" << icomb.get_nroots() << std::endl;
            exit(1); 
         }
#ifndef SERIAL
         if(icomb.world.rank() != 0){
            std::cout << "error: this function is only a serial version!" << std::endl;
            exit(1);
         }
#endif

         const double cutoff = 0.0;
         const int noff = (nsample+9)/10;
         // In case CTNS is not normalized 
         double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
         std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
         // start sampling
         double Sd = 0.0, Sd2 = 0.0, IPR = 0.0, IPR2 = 0.0;
         std::map<statetype,int> pop;
         std::map<statetype,Tm> coeff;
         auto t0 = tools::get_time();
         for(int i=0; i<nsample; i++){
            auto pr = rcanon_random(icomb,iroot);
            auto state = pr.first;
            auto ci2 = std::norm(pr.second);
            // statistical analysis
            pop[state] += 1;
            coeff[state] = pr.second;
            double s = (ci2 < cutoff)? 0.0 : -log(ci2)*ovlp;
            double ipr = ci2*ovlp;
            double fac = 1.0/(i+1.0);
            Sd   = (Sd*i + s)*fac;
            Sd2  = (Sd2*i + s*s)*fac;
            IPR  = (IPR*i + ipr)*fac;
            IPR2 = (IPR2*i + ipr*ipr)*fac; 
            if((i+1)%noff == 0){
               double std1 = std::sqrt(std::abs(Sd2-Sd*Sd)/(i+1.e-10)); // use abs in case of small negative value
               double std2 = std::sqrt(std::abs(IPR2-IPR*IPR)/(i+1.e-10)); // use abs in case of small negative value
               auto t1 = tools::get_time();
               double dt = tools::get_duration(t1-t0);
               std::cout << " i=" << std::setw(8) << i 
                  << " Sdiag=" << std::scientific << std::setw(10) 
                  << std::setprecision(3) << Sd << " std=" << std1
                  << " IPR=" << IPR << " std=" << std2
                  << " TIMING=" << dt << " S" 
                  << std::endl;	      
               t0 = tools::get_time();
            }
         }
         std::cout << "estimated Sdiag[MC]=" << Sd << " IPR[MC]=" << IPR << std::endl;
         // print important determinants
         print_samples(statestr, ovlp, pop, coeff, nsample, pthrd, nprt, saveconfs);
         auto tf = tools::get_time();
         tools::timing("ctns::rcanon_Sdiag_sample", ti, tf);
         return Sd;
      }
        
} // ctns

#endif
