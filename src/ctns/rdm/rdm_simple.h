#ifndef RDM_SIMPLE_H
#define RDM_SIMPLE_H

#include "../oper_dot_local.h"
#include "../../core/simplerdm.h"

namespace ctns{

   // Abelian case
   template <typename Qm, typename Tm>
      comb<Qm,Tm> apply_opC(const comb<Qm,Tm>& icomb,
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
         auto t0 = tools::get_time();
         std::cout << "\nctns::rdm1_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // rdm1[i,j] = <i^+j>
         linalg::matrix<Tm> rdm1(sorb,sorb);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) collapse(2)
#endif
         for(int i=0; i<sorb; i++){
            for(int j=0; j<sorb; j++){
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               auto icomb2j = apply_opC(icomb2, kj, spin_j, 0); // aj|psi2>
               auto icomb2ij = apply_opC(icomb2j, ki, spin_i, 1); // ai^+aj|psi2>
               auto smat = get_Smat(icomb1,icomb2ij); // <psi1|ai^+aj|psi2>
               // map back to the actual orbital      
               int pi = 2*image1[ki] + spin_i;
               int pj = 2*image1[kj] + spin_j;
#ifdef _OPENMP
#pragma omp critical
#endif
               rdm1(pi,pj) = smat(iroot1,iroot2);
            } // j
         } // i
         auto t1 = tools::get_time();
         tools::timing("ctns::rdm1_simple", t0, t1);
         return rdm1;
      }

   template <typename Qm, typename Tm>
      linalg::matrix<Tm> rdm2_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         auto t0 = tools::get_time();
         std::cout << "\nctns::rdm2_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // rdm2[i,j,k,l] = <i^+j^+kl> (i>j,k<l)
         int sorb2 = sorb*(sorb-1)/2;
         linalg::matrix<Tm> rdm2(sorb2,sorb2);
         for(size_t ij=0; ij<sorb2; ij++){
            auto ijpr = tools::inverse_pair0(ij);
            int i = ijpr.first;
            int j = ijpr.second;
            std::cout << "ij/pair=" << ij << "," << sorb2 << " i,j=" << i << "," << j << std::endl;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(size_t lk=0; lk<sorb2; lk++){
               auto lkpr = tools::inverse_pair0(lk);
               int l = lkpr.first;
               int k = lkpr.second; 
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               int kk = k/2, spin_k = k%2;
               int kl = l/2, spin_l = l%2;
               auto icomb2l = apply_opC(icomb2, kl, spin_l, 0); // al|psi2>
               auto icomb2kl = apply_opC(icomb2l, kk, spin_k, 0); // akal|psi2>
               auto icomb2jkl = apply_opC(icomb2kl, kj, spin_j, 1); // aj^+akal|psi2>
               auto icomb2ijkl = apply_opC(icomb2jkl, ki, spin_i, 1); // ai^+aj^+akal|psi2>
               auto smat = get_Smat(icomb1,icomb2ijkl); // <psi1|ai^+aj^+akal|psi2>
               // map back to the actual orbital      
               int pi = 2*image1[ki] + spin_i;
               int pj = 2*image1[kj] + spin_j;
               int pk = 2*image1[kk] + spin_k;
               int pl = 2*image1[kl] + spin_l;
               auto pij = tools::canonical_pair0(pi,pj);
               auto pkl = tools::canonical_pair0(pk,pl);
               Tm sgn1 = (pi>pj)? 1 : -1;
               Tm sgn2 = (pk<pl)? 1 : -1;
#ifdef _OPENMP
#pragma omp critical
#endif
               rdm2(pij,pkl) = sgn1*sgn2*smat(iroot1,iroot2);
            } // lk
         } // ij
         if(debug){
            auto rdm1 = rdm1_simple(icomb1,icomb2,iroot1,iroot2);
            auto rdm1b = fock::get_rdm1_from_rdm2(rdm2);
            auto diff = (rdm1b-rdm1).normF();
            std::cout << "Check |rdm1b-rdm1|=" << diff << std::endl;
            assert(diff < 1.e-6);
         }
         auto t1 = tools::get_time();
         tools::timing("ctns::rdm2_simple", t0, t1);
         return rdm2;
      }

   // single-site entropy
   template <typename Qm, typename Tm>
      std::vector<double> entropy1_simple(const comb<Qm,Tm>& icomb1,
            const int iroot1,
            const bool debug=true){
         std::cout << "\nctns::entropy1_simple: iroot1=" << iroot1
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int norb = icomb1.get_nphysical();
         std::vector<double> sp(norb);
         auto icomb2 = icomb1;
         for(int i=0; i<norb; i++){
            auto csite = icomb2.sites[norb-1-i];
            Tm na, nb, nanb;
            { 
               auto opBaa = get_dot_opB<Tm>(Qm::isym, 0, 0);
               icomb2.sites[norb-1-i] = contract_qt3_qt2("c", csite, opBaa);
               auto smat = get_Smat(icomb1,icomb2); 
               na = smat(iroot1,iroot1); 
            }
            {
               auto opBbb = get_dot_opB<Tm>(Qm::isym, 1, 1);
               icomb2.sites[norb-1-i] = contract_qt3_qt2("c", csite, opBbb);
               auto smat = get_Smat(icomb1,icomb2); 
               nb = smat(iroot1,iroot1); 
            }
            {
               auto opDabba = get_dot_opFabba<Tm>(Qm::isym);
               icomb2.sites[norb-1-i] = contract_qt3_qt2("c", csite, opDabba);
               auto smat = get_Smat(icomb1,icomb2);
               nanb = smat(iroot1,iroot1);
            }
            icomb2.sites[norb-1-i] = std::move(csite);
            // {|vac>,|up>,|dw>,|up,dw>}
            std::vector<double> lambda(4);
            lambda[3] = std::real(nanb); 
            lambda[2] = std::real(nb - nanb); // <dw+dw>=<nb*(1-na)>=<nb>-<nanb> 
            lambda[1] = std::real(na - nanb);
            lambda[0] = std::real(1.0 - na - nb + nanb);
            int pi = image1[i];
            sp[pi] = fock::entropy(lambda); 
         }
         if(debug){
            double sum = 0.0;
            for(int i=0; i<norb; i++){
               std::cout << " i=" << i
                  << " Sp=" << std::fixed << std::setprecision(12) << sp[i] 
                  << std::endl;
               sum += sp[i];
            }
            std::cout << "sum=" << std::setprecision(12) << sum << std::endl;
         }
         return sp;
      }

} // ctns

#endif
