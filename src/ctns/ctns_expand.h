#ifndef CTNS_EXPAND_H
#define CTNS_EXPAND_H

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "../core/csf.h"
#include "ctns_cicoeff.h"
#include "ctns_ova.h" // debug

namespace ctns{

   // --- Abelian MPS ---

   // expand CTNS into det
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      fock::onspace get_onspace(const comb<Qm,Tm>& icomb,
            const int iroot){
         std::cout << "ctns::get_onspace:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << std::endl; 
         // setup FCI space
         qsym sym_state = icomb.get_qsym_state();
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
         std::cout << "ks=" << ks << " sym=" << sym_state << " dimFCI[det]=" << dim << std::endl;
         return fci_space;
      }

   // expand CTNS into determinants
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      std::pair<fock::onspace,std::vector<Tm>> rcanon_expand_onspace(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double pthrd=1.e-2){
         std::cout << "ctns::rcanon_expand_onspace:"
            << " ifab=" << Qm::ifabelian
            << " iroot=" << iroot 
            << " pthrd=" << pthrd
            << std::endl;
         // setup FCI space
         auto fci_space = get_onspace(icomb, iroot);
         // compute exact coefficients <n|CTNS>
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         std::vector<double> pop(dim,0.0);
         double ovlp = 0.0;
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
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
            if(pop[idx] < pthrd) break;
            const auto& state = fci_space[idx];
            std::cout << " i=" << i << " idx=" << idx
               << " state=" << state
               << " pop=" << pop[idx]
               << " coeff=" << coeff[idx] 
               << std::endl;
         }
         return std::make_pair(fci_space,coeff); 
      }

} // ctns

#endif
