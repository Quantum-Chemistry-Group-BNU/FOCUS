#ifndef FCI_RDM_H
#define FCI_RDM_H

#include <vector>
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/onspace.h"
#include "../core/simplerdm.h"
#include "fci_util.h"

namespace fci{

   // <Psi1|p^+q|Psi2>
   template <typename Tm>
      void get_rdm1(const fock::onspace& space,
            const std::vector<Tm>& civec1,
            const std::vector<Tm>& civec2,
            linalg::matrix<Tm>& rdm1,
            const bool debug = false){
         const bool Htype = tools::is_complex<Tm>();
         auto t0 = tools::get_time();
         std::cout << "\nfci:get_rdm1 rdm1.shape=" << rdm1.rows() << "," << rdm1.cols() << std::endl;
         // initialization 
         rdm1 = 0.0;
         // setup product_space
         product_space pspace;
         pspace.get_pspace(space);
         // setupt coupling_table
         coupling_table ctabA, ctabB;
         ctabA.get_Cmn(pspace.spaceA, Htype);
         ctabB.get_Cmn(pspace.spaceB, Htype);
         // diagonal term: c1[i]<Di|p^+p|Di>c2[i] (i=j)
         for(size_t i=0; i<space.size(); i++){
            std::vector<int> olst;
            space[i].get_olst(olst);
            for(int p : olst){
               rdm1(p,p) += tools::conjugate(civec1[i])*civec2[i];
            }
         }
         // off-diagonal term:
         for(int ia=0; ia<pspace.dimA; ia++){
            for(const auto& pib : pspace.rowA[ia]){
               int ib = pib.first;
               int i = pib.second;
               // 1. <I_A,I_B|pA^+qA|J_A,J_B> ~ <I_A|pA^+qA|J_A><I_B|J_B> 
               //    essentially follow from construction of H_connect
               for(const auto& pja : pspace.colB[ib]){ 
                  int ja = pja.first;
                  int j = pja.second;
                  if(j <= i) continue;
                  auto search = ctabA.C11[ia].find(ja);
                  if(search != ctabA.C11[ia].end()){
                     int p[1], q[1];
                     space[i].diff_orb(space[j],p,q);
                     double sgn = space[i].parity(p[0])*space[j].parity(q[0]);
                     rdm1(p[0],q[0]) += sgn*tools::conjugate(civec1[i])*civec2[j];
                     rdm1(q[0],p[0]) += sgn*tools::conjugate(civec1[j])*civec2[i];
                  }
               } // ja
                 // 2. <I_A,I_B|pB^+qB|J_A,J_B> ~ <I_A|J_A><I_B|pB^+qB|J_B>
               for(const auto& pjb : pspace.rowA[ia]){
                  int jb = pjb.first;
                  int j = pjb.second;
                  if(j <= i) continue; 
                  auto search = ctabB.C11[ib].find(jb);
                  if(search != ctabB.C11[ib].end()){
                     int p[1], q[1];
                     space[i].diff_orb(space[j],p,q);
                     double sgn = space[i].parity(p[0])*space[j].parity(q[0]);
                     rdm1(p[0],q[0]) += sgn*tools::conjugate(civec1[i])*civec2[j];
                     rdm1(q[0],p[0]) += sgn*tools::conjugate(civec1[j])*civec2[i];
                  }
               } // jb
            } // ib
         } // ia
         if(Htype){
            // off-diagonal term: spin-flip case
            for(int ia=0; ia<pspace.dimA; ia++){
               for(const auto& pib : pspace.rowA[ia]){
                  int ib = pib.first;
                  int i = pib.second;
                  // 3. <I_A,I_B|pA^+qB|J_A,J_B> ~ <I_A|pA^+|J_A><I_B|qB|J_B> 
                  for(int ja : ctabA.C10[ia]){
                     for(const auto& pjb : pspace.rowA[ja]){ 
                        int jb = pjb.first;
                        int j = pjb.second;
                        if(j <= i) continue;
                        auto search = ctabB.C01[ib].find(jb);
                        if(search != ctabB.C01[ib].end()){
                           int p[1], q[1];
                           space[i].diff_orb(space[j],p,q);
                           double sgn = space[i].parity(p[0])*space[j].parity(q[0]);
                           rdm1(p[0],q[0]) += sgn*tools::conjugate(civec1[i])*civec2[j];
                           rdm1(q[0],p[0]) += sgn*tools::conjugate(civec1[j])*civec2[i];
                        }
                     } // jb
                  } // ja
                    // 4. <I_A,I_B|pB^+qA|J_A,J_B> ~ <I_A|qA|J_A><I_B|pB^+|J_B>
                  for(int ja : ctabA.C01[ia]){
                     for(const auto& pjb : pspace.rowA[ja]){ 
                        int jb = pjb.first;
                        int j = pjb.second;
                        if(j <= i) continue; 
                        auto search = ctabB.C10[ib].find(jb);
                        if(search != ctabB.C10[ib].end()){
                           int p[1], q[1];
                           space[i].diff_orb(space[j],p,q);
                           double sgn = space[i].parity(p[0])*space[j].parity(q[0]);
                           rdm1(p[0],q[0]) += sgn*tools::conjugate(civec1[i])*civec2[j];
                           rdm1(q[0],p[0]) += sgn*tools::conjugate(civec1[j])*civec2[i];
                        }
                     } // jb
                  } // ja
               } // ib
            } // ia
         } // Htype
         Tm tr = rdm1.trace();
         std::cout << "tr(rdm1)=" << tr << " normalized to N" << std::endl;
         auto t1 = tools::get_time();
         tools::timing("fci:get_rdm1", t0, t1);
         // debug by comparing against the brute-force implementation 
         if(debug){
            std::cout << "\ndebug fci::get_rdm1 ..." << std::endl;
            linalg::matrix<Tm> rdm1b(rdm1.rows(),rdm1.cols());
            fock::get_rdm1(space, civec1, civec2, rdm1b);
            std::cout << "tr(rdm1)=" << rdm1b.trace() << std::endl;
            auto rdm1_diff = (rdm1b - rdm1).normF();
            std::cout << "rdm1_diff=" << std::setprecision(12) << rdm1_diff << std::endl;
            if(rdm1_diff>1.e-8) tools::exit("error: difference is larger than thresh!");
         }
      }

   // <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1) using sparseH
   // which contains the computed connection information  
   template <typename Tm>
      void get_rdm2(const sparse_hamiltonian<Tm>& sparseH,
            const fock::onspace& space,
            const std::vector<Tm>& civec1,
            const std::vector<Tm>& civec2,
            linalg::matrix<Tm>& rdm2,
            const bool debug = false){
         auto t0 = tools::get_time();
         std::cout << "\nfci:get_rdm2 rdm2.shape=" << rdm2.rows() << ","<< rdm2.cols() << std::endl;
         // initialization 
         rdm2 = 0.0;
         // connected  
         int k = space[0].size();
         for(int i=0; i<sparseH.dim; i++){
            // diagonal term
            std::vector<int> olst;
            space[i].get_olst(olst);
            for(int idx=0; idx<olst.size(); idx++){
               int p0 = olst[idx]; 
               for(int jdx=0; jdx<idx; jdx++){
                  int p1 = olst[jdx];
                  int p01 = tools::canonical_pair0(p0,p1);
                  rdm2(p01,p01) += tools::conjugate(civec1[i])*civec2[i]; 
               }
            }
            // off-diagonal term: ci*<Di|p0^+p1^+q1q0|Dj>cj (j != i)
            for(int jdx=0; jdx<sparseH.connect[i].size(); jdx++){
               int j = sparseH.connect[i][jdx];
               long ph = sparseH.diff[i][jdx];
               double sgn0 = ph>0? 1 : -1;
               ph = std::abs(ph);
               int p0 = ph%k;
               int q0 = (ph/k)%k;
               // single excitations
               if(ph/k/k == 0){
                  for(const int& p1 : olst){
                     if(p1 == p0) continue;
                     int p01 = tools::canonical_pair0(p0,p1);
                     int q01 = tools::canonical_pair0(q0,p1);
                     double sgn = ((p0<p1)^(q0<p1))? -sgn0 : sgn0; 
                     rdm2(p01,q01) += sgn*tools::conjugate(civec1[i])*civec2[j];
                     rdm2(q01,p01) += sgn*tools::conjugate(civec1[j])*civec2[i];
                  }
                  // double excitations   
               }else{
                  int p1 = (ph/k/k)%k;
                  int q1 = ph/k/k/k;
                  int p01 = p0*(p0-1)/2+p1;
                  int q01 = q0*(q0-1)/2+q1;
                  rdm2(p01,q01) += sgn0*tools::conjugate(civec1[i])*civec2[j];
                  rdm2(q01,p01) += sgn0*tools::conjugate(civec1[j])*civec2[i];
               }
            }	// Dj     
         } // Di
         Tm tr = 2.0*rdm2.trace();
         std::cout << "tr(rdm2)=" << tr << " normalized to N(N-1)" << std::endl;
         auto t1 = tools::get_time();
         tools::timing("fci:get_rdm2", t0, t1);
         // debug by comparing against the brute-force implementation 
         if(debug){
            std::cout << "\ndebug fci::get_rdm2 ..." << std::endl;
            linalg::matrix<Tm> rdm2b(rdm2.rows(),rdm2.cols());
            fock::get_rdm2(space, civec1, civec2, rdm2b);
            std::cout << "tr(rdm2)=" << 2.0*rdm2.trace() << std::endl;
            auto rdm2_diff = (rdm2b - rdm2).normF();
            std::cout << "rdm2_diff=" << std::setprecision(12) << rdm2_diff << std::endl;
            if(rdm2_diff>1.e-8) tools::exit("error: difference is larger than thresh!");
         }
      }

   template <typename Tm>
      Tm get_rdm12(const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const int iroot,
            const int jroot,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            linalg::matrix<Tm>& rdm1,
            linalg::matrix<Tm>& rdm2,
            const std::string scratch,
            const bool debug = false){
         const bool Htype = tools::is_complex<Tm>();
         std::cout << "\nfci:get_rdm12 Htype=" << Htype << " k=" << rdm1.rows() << std::endl;
         auto t0 = tools::get_time();
         size_t dim = space.size();
         std::vector<Tm> civec1(vs.col(iroot), vs.col(iroot)+dim);
         std::vector<Tm> civec2(vs.col(jroot), vs.col(jroot)+dim);
         fci::sparse_hamiltonian<Tm> sparseH;
         sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype);
         fci::get_rdm1(space, civec1, civec2, rdm1, debug);
         fci::get_rdm2(sparseH, space, civec1, civec2, rdm2, debug);
         Tm Hij;
         if(iroot == jroot){
            Hij = fock::get_etot(rdm2, rdm1, int2e, int1e, ecore);
         }else{
            Hij = fock::get_etot(rdm2, rdm1, int2e, int1e, 0.0);
         }
         std::cout << "\nCheck: I,J=" << iroot << "," << jroot
            << " H(I,J)=" << std::fixed << std::setprecision(12) << Hij 
            << std::endl;
         rdm1.save_txt(scratch+"/rdm1ci."+std::to_string(iroot)+"."+std::to_string(jroot),12);
         rdm2.save_txt(scratch+"/rdm2ci."+std::to_string(iroot)+"."+std::to_string(jroot),12);
         auto t1 = tools::get_time();
         tools::timing("fci:get_rdm12", t0, t1);
         return Hij;
      }

} // fci

#endif
