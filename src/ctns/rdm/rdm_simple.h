#ifndef CTNS_RDM_SIMPLE_H
#define CTNS_RDM_SIMPLE_H

#include "../oper_dot.h"
#include "../../core/simplerdm.h"

namespace ctns{

   // Abelian case
   template <typename Qm, typename Tm>
      comb<Qm,Tm> apply_cop(const comb<Qm,Tm>& icomb,
            const int ki, 
            const int ispin, 
            const int type){
         auto icomb_i = icomb;
         // change sign
         int norb = icomb.get_nphysical();
         for(int i=0; i<ki; i++){
            icomb_i.sites[norb-1-i].mid_signed();
         }
         // apply the central operator to the site
         auto& csite = icomb_i.sites[norb-1-ki];
         auto op = get_dot_opC<Tm>(Qm::isym, ispin); // opC
         if(type == 0) op = op.H(); // opA
         csite = contract_qt3_qt2("c", csite, op);
         return icomb_i;
      }

   template <typename Qm, typename Tm>
      linalg::matrix<Tm> rdm1_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2){
         std::cout << "\nctns::rdm1_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         int sorb = 2*icomb2.get_nphysical();
         // rdm1[i,j] = <i^+j>
         linalg::matrix<Tm> rdm1(sorb,sorb);
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               auto icomb2j = apply_cop(icomb2, kj, spin_j, 0); // aj|psi2>
               auto icomb2ij = apply_cop(icomb2j, ki, spin_i, 1); // ai^+aj|psi2>
               auto smat = get_Smat(icomb1,icomb2ij); // <psi1|ai^+aj|psi2>
                                                      // map back to the actual orbital      
               int pi = 2*icomb2.topo.image2[ki] + spin_i;
               int pj = 2*icomb2.topo.image2[kj] + spin_j;
               rdm1(pi,pj) = smat(iroot1,iroot2);
            } // j
         } // i
         return rdm1;
      }

   template <typename Qm, typename Tm>
      linalg::matrix<Tm> rdm2_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "\nctns::rdm2_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         int sorb = 2*icomb2.get_nphysical();
         // rdm2[i,j,k,l] = <i^+j^+kl> (i>j,k<l)
         int sorb2 = sorb*(sorb-1)/2;
         linalg::matrix<Tm> rdm2(sorb2,sorb2);
         for(int i=0; i<sorb; i++){
            for(int j=0; j<i; j++){
               for(int l=0; l<sorb; l++){
                  for(int k=0; k<l; k++){
                     int ki = i/2, spin_i = i%2;
                     int kj = j/2, spin_j = j%2;
                     int kk = k/2, spin_k = k%2;
                     int kl = l/2, spin_l = l%2;
                     auto icomb2l = apply_cop(icomb2, kl, spin_l, 0); // al|psi2>
                     auto icomb2kl = apply_cop(icomb2l, kk, spin_k, 0); // akal|psi2>
                     auto icomb2jkl = apply_cop(icomb2kl, kj, spin_j, 1); // aj^+akal|psi2>
                     auto icomb2ijkl = apply_cop(icomb2jkl, ki, spin_i, 1); // ai^+aj^+akal|psi2>
                     auto smat = get_Smat(icomb1,icomb2ijkl); // <psi1|ai^+aj^+akal|psi2>
                     // map back to the actual orbital      
                     int pi = 2*icomb2.topo.image2[ki] + spin_i;
                     int pj = 2*icomb2.topo.image2[kj] + spin_j;
                     int pk = 2*icomb2.topo.image2[kk] + spin_k;
                     int pl = 2*icomb2.topo.image2[kl] + spin_l;
                     auto pij = tools::canonical_pair0(pi,pj);
                     auto pkl = tools::canonical_pair0(pk,pl);
                     rdm2(pij,pkl) = smat(iroot1,iroot2);
                  } // l
               } // k
            } // j
         } // i
         if(debug){
            auto rdm1 = rdm1_simple(icomb1,icomb2,iroot1,iroot2);
            auto rdm1b = fock::get_rdm1_from_rdm2(rdm2);
            auto diff = (rdm1b-rdm1).normF();
            std::cout << "Check |rdm1b-rdm1|=" << diff << std::endl;
            assert(diff < 1.e-6);
         }
         return rdm2;
      }

} // ctns

#endif
