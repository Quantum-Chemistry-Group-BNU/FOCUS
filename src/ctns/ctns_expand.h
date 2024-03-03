#ifndef CTNS_EXPAND_H
#define CTNS_EXPAND_H

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "../core/csf.h"
#include "ctns_cicoeff.h"
// debug
#include "ctns_ova.h"
#include "sadmrg/ctns_csf2samps.h"

namespace ctns{

   // --- Abelian MPS ---

   // expand CTNS into det
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      fock::onspace rcanon_expand_onspace(const comb<Qm,Tm>& icomb,
            const int iroot){
         std::cout << "\nctns::rcanon_expand_onspace iroot=" << iroot << std::endl; 
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
         std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI[det]=" << dim << std::endl;
         return fci_space;
      }
 
   // expand CTNS into determinants
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      std::pair<fock::onspace,std::vector<Tm>> rcanon_expand_onstate(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         std::cout << "\nctns::rcanon_expand_onstate iroot=" << iroot 
            << " thresh_print=" << thresh_print << std::endl;
         // setup FCI space
         auto fci_space = rcanon_expand_onspace(icomb, iroot);
         // compute exact coefficients <n|CTNS>
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
         }
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2);
         double Sdiag = fock::coeff_entropy(coeff);
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl; 
         return std::make_pair(fci_space,coeff); 
      }

   // --- Non-Abelian MPS

   // expand CTNS into determinants / csf
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      fock::csfspace rcanon_expand_csfspace(const comb<Qm,Tm>& icomb,
            const int iroot){
         std::cout << "\nctns::rcanon_expand_csfspace iroot=" << iroot << std::endl; 
         // setup FCI space
         qsym sym_state = icomb.get_sym_state();
         int ne = sym_state.ne(); 
         int ks = icomb.get_nphysical();
         int ts = sym_state.ts(); 
         fock::csfspace fci_space = fock::get_csf_space(ks,ne,ts); 
         size_t dim = fci_space.size();
         std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI[csf]=" << dim << std::endl;
         return fci_space;
      }

   // expand CTNS into csf
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::csfspace,std::vector<Tm>> rcanon_expand_csfstate(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         std::cout << "\nctns::rcanon_expand_csfstate iroot=" << iroot 
            << " thresh_print=" << thresh_print << std::endl;
         // setup FCI space
         auto fci_space = rcanon_expand_csfspace(icomb, iroot);
         // compute exact coefficients <n|CTNS>
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            // method-1: 
            auto samps = csf2samps<Tm>(icomb.topo, state);
            coeff[i] = get_Smat(samps, icomb)(0,iroot);
            // method-2:
            auto tmp = rcanon_CIcoeff(icomb, state)[iroot];
            if(std::abs(coeff[i]-tmp) > 1.e-10){
               std::cout << "error: inconsistent evaluation of wfcoeff:" << std::endl;
               std::cout << "coeff by csf2samps&get_Smat=" << coeff[i] << std::endl;
               std::cout << "coeff by rcanon_CIcoeff=" << tmp << std::endl;
               std::cout << "difference=" << coeff[i]-tmp << std::endl;
               exit(1);
            }
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
         }
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2);
         double Sdiag = fock::coeff_entropy(coeff);
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;
         return std::make_pair(fci_space,coeff); 
      }

   // expand CTNS into det
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::pair<fock::onspace,std::vector<Tm>> rcanon_expand_onstate(const comb<Qm,Tm>& icomb,
            const int iroot,
            const double thresh_print=1.e-10){
         std::cout << "\nctns::rcanon_expand_onstate iroot=" << iroot 
            << " thresh_print=" << thresh_print << std::endl;
         auto csf_expansion = rcanon_expand_csfstate(icomb, iroot, thresh_print);
         const auto& csf_space = csf_expansion.first;
         const auto& csf_coeff = csf_expansion.second; 
         qsym sym_state = icomb.get_sym_state();
         int ne = sym_state.ne(); 
         int ks = icomb.get_nphysical();
         int ts = sym_state.ts(); 
         int na = (ne+ts)/2, nb = ne-na;
         fock::onspace fci_space = fock::get_fci_space(ks,na,nb); 
         size_t dim = fci_space.size();
         std::vector<Tm> coeff(dim,0.0);
         // setup map
         std::map<fock::onstate,int> idxmap;
         for(int i=0; i<dim; i++){
            idxmap[fci_space[i]] = i;
         }
         // compute det coeff
         for(int i=0; i<csf_space.size(); i++){
            const auto& csf = csf_space[i];
            auto det_expansion = csf.to_det();
            const auto& det_space = det_expansion.first;
            const auto& det_coeff = det_expansion.second;
            for(int j=0; j<det_space.size(); j++){
               const auto& det = det_space[j];
               coeff[idxmap[det]] += csf_coeff[i]*det_coeff[j];
            }
         }
         // print
         for(int i=0; i<dim; i++){
            const auto& state = fci_space[i];
            coeff[i] *= state.permute_sgn(icomb.topo.image2); // convert back to physical ordering 
            if(std::abs(coeff[i]) < thresh_print) continue;
            std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
         }
         double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2);
         double Sdiag = fock::coeff_entropy(coeff);
         std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl; 
         return std::make_pair(fci_space,coeff); 
      }

} // ctns

#endif
