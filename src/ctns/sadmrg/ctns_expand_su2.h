#ifndef CTNS_EXPAND_SU2_H
#define CTNS_EXPAND_SU2_H

#include "../../core/onspace.h"
#include "../../core/analysis.h"
#include "../../core/csf.h"
#include "../ctns_ova.h"
#include "ctns_cicoeff_su2.h"
#include "ctns_csf2samps.h"

namespace ctns{

   // --- Non-Abelian MPS

   // expand CTNS into determinants / csf
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      fock::csfspace get_csfspace(const comb<Qm,Tm>& icomb,
            const int iroot,
            const int iprt=1){
         if(iprt > 0){
            std::cout << "ctns::get_csfspace:"
               << " ifab=" << Qm::ifabelian
               << " iroot=" << iroot 
               << std::endl; 
         }
         // setup FCI space
         qsym sym_state = icomb.get_qsym_state();
         int ne = sym_state.ne(); 
         int ks = icomb.get_nphysical();
         int ts = sym_state.ts(); 
         fock::csfspace fci_space = fock::get_csf_space(ks,ne,ts); 
         size_t dim = fci_space.size();
         if(iprt > 0){
            std::cout << "ks=" << ks << " sym=" << sym_state << " dimFCI[csf]=" << dim << std::endl;
         }
         return fci_space;
      }

   // expand CTNS into csf
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::csfspace,std::vector<Tm>> rcanon_expand_csfspace(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double pthrd=1.e-2,
            const int iprt=1){
         if(iprt>0){
            std::cout << "ctns::rcanon_expand_csfspace:"
               << " ifab=" << Qm::ifabelian
               << " iroot=" << iroot 
               << " pthrd=" << pthrd
               << std::endl;
         }
         // setup FCI space
         auto fci_space = get_csfspace(icomb, iroot, iprt);
         // compute exact coefficients <n|CTNS>
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         std::vector<double> pop(dim,0.0);
         double ovlp = 0.0;
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            // method-1: 
            auto samps = csf2samps<Tm>(icomb.topo, state);
            coeff[i] = get_Smat(samps, icomb)(0,iroot);
            // method-2:
            auto tmp = rcanon_CIcoeff(icomb, state)[iroot];
            if(std::abs(coeff[i]-tmp) > 1.e-10){
               std::cout << "error: inconsistent evaluation of wfcoeff:" << std::endl;
               std::cout << "coeff by csf2samps|get_Smat=" << coeff[i] << std::endl;
               std::cout << "coeff by rcanon_CIcoeff=" << tmp << std::endl;
               std::cout << "difference=" << coeff[i]-tmp << std::endl;
               exit(1);
            }
            pop[i] = std::norm(coeff[i]);
            ovlp += pop[i];
         }
         if(iprt > 0){
            std::cout << "ovlp=" << ovlp << std::endl;
            if(std::abs(ovlp-1.0)>1.e-8){
               std::cout << "error: ovlp deviates from 1! dev=" << ovlp-1.0 << std::endl;
               exit(1);
            }
            auto indx = tools::sort_index(pop,1);
            for(int i=0; i<dim; i++){
               int idx = indx[i];
               if(pop[idx] < pthrd) break;
               const auto& state = fci_space[idx];
               std::cout << " i=" << i << " idx=" << idx
                  << " state=" << state
                  << " pop=" << pop[idx]
                  << " coeff=" << coeff[idx] 
                  << std::endl;
            }
         }
         return std::make_pair(fci_space,coeff); 
      }

   // expand CTNS into det
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::onspace,std::vector<Tm>> rcanon_expand_onspace0(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double pthrd=1.e-2){
         const bool debug = false;
         std::cout << "ctns::rcanon_expand_onspace0:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " pthrd=" << pthrd
            << std::endl;
         const int iprt = 0;
         // expand icomb to csf space first
         auto csf_expansion = rcanon_expand_csfspace(icomb, iroot, pthrd, iprt);
         const auto& csf_space = csf_expansion.first;
         const auto& csf_coeff = csf_expansion.second; 
         qsym sym_state = icomb.get_qsym_state();
         int ne = sym_state.ne(), ts = sym_state.ts(), tm = ts; 
         int na = (ne+tm)/2, nb = ne-na;
         int ks = icomb.get_nphysical();
         fock::onspace fci_space = fock::get_fci_space(ks,na,nb); // onspace 
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         std::vector<double> pop2(dim,0.0);
         // setup map
         std::map<fock::onstate,int> idxmap;
         for(int i=0; i<dim; i++){
            idxmap[fci_space[i]] = i;
         }
         // compute det coeff
         for(int i=0; i<csf_space.size(); i++){
            const auto& csf = csf_space[i];
            if(debug) std::cout << "i=" << i << " csf=" << csf << std::endl;
            auto det_expansion = csf.to_det();
            const auto& det_space = det_expansion.first;
            const auto& det_coeff = det_expansion.second;
            for(int j=0; j<det_space.size(); j++){
               const auto& det = det_space[j];
               coeff[idxmap[det]] += csf_coeff[i]*det_coeff[j];
               pop2[idxmap[det]] += std::norm(csf_coeff[i]*det_coeff[j]);
            }
         }
         // print
         std::vector<double> pop(dim,0.0);
         double ovlp = 0.0;
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            // ZL@20240829: Use the rindex ordering, without converting to natural ordering 0123...
            //coeff[i] *= state.permute_sgn(icomb.topo.image2); // convert back to physical ordering 
            pop[i] = std::norm(coeff[i]);
            ovlp += pop[i];
         }
         std::cout << "ovlp=" << ovlp << std::endl;
         if(std::abs(ovlp-1.0)>1.e-8){
            std::cout << "error: ovlp deviates from 1! dev=" << ovlp-1.0 << std::endl;
            exit(1);
         }
         auto indx = tools::sort_index(pop,1);
         for(int i=0; i<dim; i++){  
            int idx = indx[i];
            if(pop[idx] < pthrd) continue;
            const auto& state = fci_space[idx];
            std::cout << " i=" << i << " idx=" << idx
               << " state=" << state
               << " pop=" << pop[idx]
               << " coeff=" << coeff[idx] 
               << std::endl;
         }
         return std::make_pair(fci_space,coeff); 
      }

   // expand CTNS into csf
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::onspace,std::vector<Tm>> rcanon_expand_onspace(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double pthrd=1.e-2,
            const int iprt=1){
         if(iprt>0){
            std::cout << "ctns::rcanon_expand_onspace:"
               << " ifab=" << Qm::ifabelian
               << " iroot=" << iroot 
               << " pthrd=" << pthrd
               << std::endl;
         }
         qsym sym_state = icomb.get_qsym_state();
         int ne = sym_state.ne(), ts = sym_state.ts(), tm = ts;
         int na = (ne+tm)/2, nb = ne-na;
         assert(na >= 0 and nb >= 0);
         int ks = icomb.get_nphysical();
         auto fci_space = fock::get_fci_space(ks,na,nb);
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         // compute exact coefficients <n|CTNS>
         std::vector<double> pop(dim,0.0);
         double ovlp = 0.0;
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
            pop[i] = std::norm(coeff[i]);
            ovlp += pop[i];
         }
         if(iprt > 0){
            std::cout << "ovlp=" << ovlp << std::endl;
            if(std::abs(ovlp-1.0)>1.e-8){
               std::cout << "error: ovlp deviates from 1! dev=" << ovlp-1.0 << std::endl;
               exit(1);
            }
            auto indx = tools::sort_index(pop,1);
            for(int i=0; i<dim; i++){
               int idx = indx[i];
               if(pop[idx] < pthrd) break;
               const auto& state = fci_space[idx];
               std::cout << " i=" << i << " idx=" << idx
                  << " state=" << state
                  << " pop=" << pop[idx]
                  << " coeff=" << coeff[idx] 
                  << std::endl;
            }
         }
         const bool debug = true;
         if(debug){
            auto result = rcanon_expand_onspace0(icomb, iroot, pthrd);
            assert(result.first.size() == fci_space.size());
            linalg::xaxpy(coeff.size(), -1.0, coeff.data(), result.second.data());
            auto diff = linalg::xnrm2(coeff.size(), result.second.data());
            std::cout << "debug: |v-v0|=" << diff << std::endl;
            assert(diff < 1.e-10);
         }
         return std::make_pair(fci_space,coeff); 
      }

} // ctns

#endif
