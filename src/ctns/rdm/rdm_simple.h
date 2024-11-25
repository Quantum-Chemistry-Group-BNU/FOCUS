#ifndef RDM_SIMPLE_H
#define RDM_SIMPLE_H

#include "../oper_dot_local.h"
#include "../../core/simplerdm.h"
#include "../ctns_ova.h"

namespace ctns{

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      comb<Qm,Tm> apply_opC(const comb<Qm,Tm>& icomb,
            const int ki, 
            const int ispin, 
            const int type){
         std::cout << "error: apply_opC does not support su2 case!" << std::endl;
         exit(1);
      }
   //
   // Abelian case
   //
   //      Sgn Sgn Sgn OpC  Id  Id
   //       |   |   |   |   |   |
   //  --*--*---*---*---*---*---*
   //
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      comb<Qm,Tm> apply_opC(const comb<Qm,Tm>& icomb,
            const int ki, 
            const int ispin, 
            const int type){
         // copy
         auto icomb_i = icomb;
         // change sign
         auto sym_op = type==1? get_qsym_opC(Qm::isym, ispin) : get_qsym_opD(Qm::isym, ispin); 
         int norb = icomb.get_nphysical();
         for(int i=0; i<ki; i++){
            auto& site = icomb_i.sites[norb-1-i];
            site.mid_signed();
            site.info.qrow.add(sym_op);
            site.info.qcol.add(sym_op);
         }
         // change qsym
         for(int i=0; i<icomb_i.get_nroots(); i++){
            auto& rwf = icomb_i.rwfuns[i];
            rwf.info.qrow.add(sym_op);
            rwf.info.qcol.add(sym_op);   
         }
         // apply the central operator to the site
         auto& csite = icomb_i.sites[norb-1-ki];
         auto op = get_dot_opC<Tm>(Qm::isym, ispin); // opC
         if(type == 0) op = op.H(); // opA
         csite = contract_qt3_qt2("c", csite, op);
         csite.info.sym = qsym(Qm::isym,0,0);
         csite.info.qrow.add(sym_op);
         // canonicalize last dot to identity, otherwise 
         // operator construction in rdm_env will fail.
         rcanon_lastdots(icomb_i);
         return icomb_i;
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rdm1_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2){
         std::cout << "error: rdm1_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
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

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rdm2_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "error: rdm2_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
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
         // rdm2[ij,lk] = <i^+j^+kl> (i>j,k<l)
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
               auto plk = tools::canonical_pair0(pl,pk);
               Tm sgn1 = tools::sgn_pair0(pi,pj);
               Tm sgn2 = tools::sgn_pair0(pl,pk);
#ifdef _OPENMP
#pragma omp critical
#endif
               rdm2(pij,plk) = sgn1*sgn2*smat(iroot1,iroot2);
            } // lk
         } // ij
         if(debug){
            int nelec2 = icomb2.get_qsym_state().ne();
            auto rdm1 = rdm1_simple(icomb1,icomb2,iroot1,iroot2);
            auto rdm1b = fock::get_rdm1_from_rdm2(rdm2,true,nelec2);
            auto diff = (rdm1b-rdm1).normF();
            std::cout << "Check |rdm1b-rdm1|=" << diff << std::endl;
            assert(diff < 1.e-6);
         }
         auto t1 = tools::get_time();
         tools::timing("ctns::rdm2_simple", t0, t1);
         return rdm2;
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rdm3_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "error: rdm3_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rdm3_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=true){
         auto t0 = tools::get_time();
         std::cout << "\nctns::rdm3_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // rdm3[ijk,nml] = <i^+j^+k^+lmn> (i>j>k,l<m<n)
         size_t sorb3 = sorb*(sorb-1)*(sorb-2)/6;
         linalg::matrix<Tm> rdm3(sorb3,sorb3);
         for(size_t ijk=0; ijk<sorb3; ijk++){
            auto ijktr = tools::inverse_triple0(ijk);
            int i = std::get<0>(ijktr);
            int j = std::get<1>(ijktr);
            int k = std::get<2>(ijktr);
            std::cout << "ijk/triple=" << ijk << "," << sorb3 << " i,j,k=" << i << "," << j << "," << k << std::endl;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(size_t nml=0; nml<sorb3; nml++){
               auto nmltr = tools::inverse_triple0(nml);
               int n = std::get<0>(nmltr);
               int m = std::get<1>(nmltr); 
               int l = std::get<2>(nmltr); 
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               int kk = k/2, spin_k = k%2;
               int kl = l/2, spin_l = l%2;
               int km = m/2, spin_m = m%2;
               int kn = n/2, spin_n = n%2;
               auto icomb2n = apply_opC(icomb2, kn, spin_n, 0); // an|psi2>
               auto icomb2mn = apply_opC(icomb2n, km, spin_m, 0); // aman|psi2>
               auto icomb2lmn = apply_opC(icomb2mn, kl, spin_l, 0); // alaman|psi2>
               auto icomb2klmn = apply_opC(icomb2lmn, kk, spin_k, 1); // ak+alaman|psi2>
               auto icomb2jklmn = apply_opC(icomb2klmn, kj, spin_j, 1); // aj+ak+alaman|psi2>
               auto icomb2ijklmn = apply_opC(icomb2jklmn, ki, spin_i, 1); // ai+aj+ak+alaman|psi2>
               auto smat = get_Smat(icomb1,icomb2ijklmn); // <psi1|ai+aj+ak+alaman|psi2>
               // map back to the actual orbital      
               int pi = 2*image1[ki] + spin_i;
               int pj = 2*image1[kj] + spin_j;
               int pk = 2*image1[kk] + spin_k;
               int pl = 2*image1[kl] + spin_l;
               int pm = 2*image1[km] + spin_m;
               int pn = 2*image1[kn] + spin_n;
               auto pijk = tools::canonical_triple0(pi,pj,pk);
               auto pnml = tools::canonical_triple0(pn,pm,pl);
               Tm sgn1 = tools::sgn_triple0(pi,pj,pk);
               Tm sgn2 = tools::sgn_triple0(pn,pm,pl);
#ifdef _OPENMP
#pragma omp critical
#endif
               rdm3(pijk,pnml) = sgn1*sgn2*smat(iroot1,iroot2);
            } // lmn
         } // ijk
         if(debug){
            int nelec2 = icomb2.get_qsym_state().ne();
            auto rdm2 = rdm2_simple(icomb1,icomb2,iroot1,iroot2);
            auto rdm2b = fock::get_rdm2_from_rdm3(rdm3,true,nelec2);
            auto diff2 = (rdm2b-rdm2).normF();
            std::cout << "\nCheck |rdm2b-rdm2|=" << diff2 << std::endl;
            auto rdm1 = fock::get_rdm1_from_rdm2(rdm2,true,nelec2);
            auto rdm1b = fock::get_rdm1_from_rdm2(rdm2b,true,nelec2);
            auto diff1 = (rdm1b-rdm1).normF();
            std::cout << "\nCheck |rdm1b-rdm1|=" << diff1 << std::endl;
            assert(diff2 < 1.e-6 and diff1 < 1.e-6);
         }
         auto t1 = tools::get_time();
         tools::timing("ctns::rdm3_simple", t0, t1);
         return rdm3;
      }

   // single-site entropy
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::vector<double> entropy1_simple(const comb<Qm,Tm>& icomb1,
            const int iroot1,
            const bool debug=true){
         std::cout << "error: entropy1_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      std::vector<double> entropy1_simple(const comb<Qm,Tm>& icomb1,
            const int iroot1,
            const bool debug=true){
         std::cout << "\nctns::entropy1_simple: iroot1=" << iroot1
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int norb = icomb1.get_nphysical();
         std::vector<double> sp(norb);
         std::vector<double> lambda(4*norb);
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
            lambda[4*i+3] = std::real(nanb); 
            lambda[4*i+2] = std::real(nb - nanb); // <dw+dw>=<nb*(1-na)>=<nb>-<nanb> 
            lambda[4*i+1] = std::real(na - nanb);
            lambda[4*i+0] = std::real(1.0 - na - nb + nanb);
            std::vector<double> lambda_i(4);
            lambda_i[0] = lambda[4*i];
            lambda_i[1] = lambda[4*i+1];
            lambda_i[2] = lambda[4*i+2];
            lambda_i[3] = lambda[4*i+3];
            int pi = image1[i];
            sp[pi] = fock::pop_entropy(lambda_i);
         }
         if(debug){
            double sum = 0.0;
            for(int i=0; i<norb; i++){
               std::cout << " i=" << i
                  << " lambda={" << std::fixed << std::setprecision(3)
                  << "0:" << lambda[4*i] << ", a:" << lambda[4*i+1] 
                  << ", b:" << lambda[4*i+2] << ", ab:" << lambda[4*i+3] << "}" 
                  << " Sp=" << std::fixed << std::setprecision(12) << sp[i] 
                  << std::endl;
               sum += sp[i];
            }
            std::cout << "sum=" << std::setprecision(12) << sum << std::endl;
         }
         return sp;
      }

   //
   // Transition density matrices
   //
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm1p0h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2){
         std::cout << "error: tdm1p0h_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm1p0h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2){
         auto t0 = tools::get_time();
         std::cout << "\nctns::tdm1p0h_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // tdm1[i,j] = <psi|i^+|psi2>
         linalg::matrix<Tm> tdm1(sorb,1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(int i=0; i<sorb; i++){
            int ki = i/2, spin_i = i%2;
            auto icomb2i = apply_opC(icomb2, ki, spin_i, 1); // ai^+|psi2>
            auto smat = get_Smat(icomb1,icomb2i); // <psi1|ai^+|psi2>
            // map back to the actual orbital      
            int pi = 2*image1[ki] + spin_i;
#ifdef _OPENMP
#pragma omp critical
#endif
            tdm1(pi,0) = smat(iroot1,iroot2);
         } // i
         auto t1 = tools::get_time();
         tools::timing("ctns::tdm1p0h_simple", t0, t1);
         return tdm1;
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm2p0h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "error: tdm2p0h_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm2p0h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         auto t0 = tools::get_time();
         std::cout << "\nctns::tdm2p0h_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // tdm2[ij] = <i^+j^+> (i>j)
         int sorb2 = sorb*(sorb-1)/2;
         linalg::matrix<Tm> tdm2(sorb2,1);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
         for(size_t ij=0; ij<sorb2; ij++){
            auto ijpr = tools::inverse_pair0(ij);
            int i = ijpr.first;
            int j = ijpr.second;
            std::cout << "ij/pair=" << ij << "," << sorb2 << " i,j=" << i << "," << j << std::endl;
            int ki = i/2, spin_i = i%2;
            int kj = j/2, spin_j = j%2;
            auto icomb2j = apply_opC(icomb2, kj, spin_j, 1); // aj^+|psi2>
            auto icomb2ij = apply_opC(icomb2j, ki, spin_i, 1); // ai^+aj^+|psi2>
            auto smat = get_Smat(icomb1,icomb2ij); // <psi1|ai^+aj^+|psi2>
            // map back to the actual orbital      
            int pi = 2*image1[ki] + spin_i;
            int pj = 2*image1[kj] + spin_j;
            auto pij = tools::canonical_pair0(pi,pj);
            Tm sgn1 = tools::sgn_pair0(pi,pj);
#ifdef _OPENMP
#pragma omp critical
#endif
            tdm2(pij,0) = sgn1*smat(iroot1,iroot2);
         } // ij
         auto t1 = tools::get_time();
         tools::timing("ctns::tdm2p0h_simple", t0, t1);
         return tdm2;
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm2p1h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "error: tdm2p1h_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm2p1h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         auto t0 = tools::get_time();
         std::cout << "\nctns::tdm2p1h_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // tdm2[ij,k] = <i^+j^+k> (i>j)
         int sorb2 = sorb*(sorb-1)/2;
         linalg::matrix<Tm> tdm2(sorb2,sorb);
         for(size_t ij=0; ij<sorb2; ij++){
            auto ijpr = tools::inverse_pair0(ij);
            int i = ijpr.first;
            int j = ijpr.second;
            std::cout << "ij/pair=" << ij << "," << sorb2 << " i,j=" << i << "," << j << std::endl;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(size_t k=0; k<sorb; k++){
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               int kk = k/2, spin_k = k%2;
               auto icomb2k = apply_opC(icomb2, kk, spin_k, 0); // ak|psi2>
               auto icomb2jk = apply_opC(icomb2k, kj, spin_j, 1); // aj^+ak|psi2>
               auto icomb2ijk = apply_opC(icomb2jk, ki, spin_i, 1); // ai^+aj^+ak|psi2>
               auto smat = get_Smat(icomb1,icomb2ijk); // <psi1|ai^+aj^+ak|psi2>
               // map back to the actual orbital
               int pi = 2*image1[ki] + spin_i;
               int pj = 2*image1[kj] + spin_j;
               int pk = 2*image1[kk] + spin_k;
               auto pij = tools::canonical_pair0(pi,pj);
               Tm sgn1 = tools::sgn_pair0(pi,pj);
#ifdef _OPENMP
#pragma omp critical
#endif
               tdm2(pij,pk) = sgn1*smat(iroot1,iroot2);
            } // k
         } // ij
         auto t1 = tools::get_time();
         tools::timing("ctns::tdm2p1h_simple", t0, t1);
         return tdm2;
      }

   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm3p2h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=false){
         std::cout << "error: tdm3p2h_simple does not support su2 case!" << std::endl;
         exit(1);
      }
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> tdm3p2h_simple(const comb<Qm,Tm>& icomb1,
            const comb<Qm,Tm>& icomb2,
            const int iroot1,
            const int iroot2,
            const bool debug=true){
         auto t0 = tools::get_time();
         std::cout << "\nctns::tdm3p2h_simple: iroot1=" << iroot1
            << " iroot2=" << iroot2 
            << std::endl;
         assert(iroot1 < icomb1.get_nroots());
         assert(iroot2 < icomb2.get_nroots());
         auto image1 = icomb1.topo.get_image1();
         int sorb = 2*icomb2.get_nphysical();
         // tdm3[ijk,ml] = <i^+j^+k^+lm> (i>j>k,l<m)
         size_t sorb3 = sorb*(sorb-1)*(sorb-2)/6;
         size_t sorb2 = sorb*(sorb-1)/2;
         linalg::matrix<Tm> tdm3(sorb3,sorb2);
         for(size_t ijk=0; ijk<sorb3; ijk++){
            auto ijktr = tools::inverse_triple0(ijk);
            int i = std::get<0>(ijktr);
            int j = std::get<1>(ijktr);
            int k = std::get<2>(ijktr);
            std::cout << "ijk/triple=" << ijk << "," << sorb3 << " i,j,k=" << i << "," << j << "," << k << std::endl;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for(size_t ml=0; ml<sorb2; ml++){
               auto mlpr = tools::inverse_pair0(ml);
               int m = mlpr.first;
               int l = mlpr.second;
               int ki = i/2, spin_i = i%2;
               int kj = j/2, spin_j = j%2;
               int kk = k/2, spin_k = k%2;
               int kl = l/2, spin_l = l%2;
               int km = m/2, spin_m = m%2;
               auto icomb2m = apply_opC(icomb2, km, spin_m, 0); // am|psi2>
               auto icomb2lm = apply_opC(icomb2m, kl, spin_l, 0); // alam|psi2>
               auto icomb2klm = apply_opC(icomb2lm, kk, spin_k, 1); // ak+alam|psi2>
               auto icomb2jklm = apply_opC(icomb2klm, kj, spin_j, 1); // aj+ak+alam|psi2>
               auto icomb2ijklm = apply_opC(icomb2jklm, ki, spin_i, 1); // ai+aj+ak+alam|psi2>
               auto smat = get_Smat(icomb1,icomb2ijklm); // <psi1|ai+aj+ak+alam|psi2>
               // map back to the actual orbital      
               int pi = 2*image1[ki] + spin_i;
               int pj = 2*image1[kj] + spin_j;
               int pk = 2*image1[kk] + spin_k;
               int pl = 2*image1[kl] + spin_l;
               int pm = 2*image1[km] + spin_m;
               auto pijk = tools::canonical_triple0(pi,pj,pk);
               auto pml = tools::canonical_pair0(pm,pl);
               Tm sgn1 = tools::sgn_triple0(pi,pj,pk);
               Tm sgn2 = tools::sgn_pair0(pm,pl);
#ifdef _OPENMP
#pragma omp critical
#endif
               tdm3(pijk,pml) = sgn1*sgn2*smat(iroot1,iroot2);
            } // lm
         } // ijk
         auto t1 = tools::get_time();
         tools::timing("ctns::tdm3p2h_simple", t0, t1);
         return tdm3;
      }

} // ctns

#endif
