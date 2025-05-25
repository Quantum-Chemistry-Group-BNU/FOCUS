#ifndef CTNS_CICOEFF_SU2_H
#define CTNS_CICOEFF_SU2_H

/*
   Algorithms for CTNS:

   2. rcanon_CIcoeff: <n|CTNS>
      rcanon_CIovlp: <CI|CTNS>
      rcanon_CIcoeff_check
*/

#include "../../core/onspace.h"
#include "../../core/analysis.h"
#include "../../core/csf.h"
#include "../../core/spin.h"
#include "../ctns_comb.h"

namespace ctns{

   // Algorithm 2:
   // <CSF|CTNS[i]>
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::vector<Tm> rcanon_CIcoeff(const comb<Qm,Tm>& icomb,
            const fock::csfstate& state){
         // only correct for MPS, because csf is linearly coupled.
         assert(icomb.topo.ifmps);
         // finally return coeff = <n|CTNS[i]> as a vector 
         int n = icomb.get_nroots(); 
         std::vector<Tm> coeff(n,0.0);
         // intermediate quantum number
         auto narray  = state.intermediate_narray();
         auto tsarray = state.intermediate_tsarray();
         
         /*         
         std::cout << "\ncsf=" << state << std::endl;
         tools::print_vector(narray,"narray");
         tools::print_vector(tsarray,"tsarray");
         //
         // csf=222000 [from right to left] 
         //
         //    5   4   3   2   1   0 
         //    |   |   |   |   |   | 
         // -<-0---0---0---2---2---2-<-
         //  6   6   6   6   4   2   0
         //
         //  narray= 6 6 6 6 4 2 0
         //  tsarray= 0 0 0 0 0 0 0
         */ 
         
         // compute <n|CTNS> by contracting all sites
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         linalg::matrix<Tm> bmat;
         // loop from the rightmost site of MPS to the left
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
            assert(i == node.lindex);
            int tp = node.type;
            if(tp == 0 || tp == 1){
               // site on backbone with physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               qsym ql(3,narray[i],tsarray[i]);
               qsym qr(3,narray[i+1],tsarray[i+1]);
               qsym qc(3,state.nvec(i),state.dvec(i)==1 or state.dvec(i)==2);
               // take out a block of data
               auto blk3 = site.get_rcf_symblk(ql,qr,qc);
               // in case this CTNS does not encode this csf, no such block 
               if(blk3.empty()) return coeff;
               int dl = blk3.dim0;
               int dr = blk3.dim1;
               // given occ pattern, extract the corresponding qblock
               if(i == icomb.topo.nbackbone-1){
                  assert(dr == 1);
                  linalg::matrix<Tm> tmp(dl,dr,blk3.data());
                  bmat = std::move(tmp); 
               }else{
                  linalg::matrix<Tm> tmp(dl,dr,blk3.data());
                  bmat = linalg::xgemm("N","N",tmp,bmat);
               }
            } // tp
         } // i
         // final contraction with rwfun
         qsym qout(3,narray[0],tsarray[0]);
         auto wf2 = icomb.get_wf2();
         int bc = wf2.info.qcol.existQ(qout); 
         auto wf2blk = wf2(0,bc);
         auto wfcoeff = linalg::xgemm("N","N",wf2blk,bmat);
         assert(wfcoeff.rows() == n && wfcoeff.cols() == 1);
         linalg::xcopy(n, wfcoeff.data(), coeff.data());
         return coeff;
      }

   // SA-MPS and CSF
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rcanon_CIovlp(const comb<Qm,Tm>& icomb,
            const fock::csfspace& space,
            const linalg::matrix<Tm>& vs,
            const bool reorder=false){
         assert(space.size() == vs.rows());
         std::cout << "\nctns::rcanon_CIovlp"
            << " ifab=" << Qm::ifabelian
            << " reorder=" << reorder
            << " cistates=" << vs.cols() 
            << " ciconfs=" << space.size() 
            << std::endl;
         if(!Qm::ifabelian and reorder){
            std::cout << "error: csf does not support reorder!" << std::endl;
            exit(1);
         }
         int n = icomb.get_nroots(); 
         size_t dim = space.size();
         // cmat[n,i] = <D[i]|CTNS[n]>
         linalg::matrix<Tm> cmat(n,dim);
         for(int i=0; i<dim; i++){
            auto state = space[i];
            auto coeff = rcanon_CIcoeff(icomb, state);
            linalg::xcopy(n, coeff.data(), cmat.col(i));
         }
         // ovlp[i,n] = vs*[k,i] cmat[n,k]
         auto ovlp = linalg::xgemm("C","T",vs,cmat);
         return ovlp;
      }

   // <n|CTNS[i]> by contracting the CTNS
   //
   // |n>=|n1n2n3>, |q> can be |q1q3q2>, |psi>=|q>*psi(q)
   // then <n|psi> = <n|q>*psi(q), <n|q> is the sign
   //
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      std::vector<Tm> rcanon_CIcoeff(const comb<Qm,Tm>& icomb,
            const fock::onstate& state){
         const bool debug = false;
         // only correct for MPS, because csf is linearly coupled.
         assert(icomb.topo.ifmps);
         // finally return coeff = <n|CTNS[i]> as a vector 
         int nsite = icomb.get_nphysical();
         int n = icomb.get_nroots(); 
         std::vector<Tm> coeff(n,0.0);
         // intermediate quantum number
         auto narray  = state.intermediate_narray();
         auto tmarray = state.intermediate_tmarray();
         if(debug){
            std::cout << "\ndet=" << state << std::endl;
            tools::print_vector(narray,"narray");
            tools::print_vector(tmarray,"tmarray");
            //
            // det=222000 [from right to left] 
            //
            //    5   4   3   2   1   0 
            //    |   |   |   |   |   | 
            // -<-0---0---0---2---2---2-<-
            //  6   6   6   6   4   2   0
            //
            //  narray= 6 6 6 6 4 2 0
            //  tsarray= 0 0 0 0 0 0 0
         }
         // check consistency
         auto sym_state = icomb.get_qsym_state();
         if(narray[0] != sym_state.ne() or std::abs(tmarray[0]) > sym_state.ts()){
            return coeff;
         }
         // compute <n|CTNS> by contracting all sites
         qbond qright;
         qright.dims = {{qsym(3,0,0),1}};
         std::vector<linalg::matrix<Tm>> bmats(1);
         bmats[0] = linalg::identity_matrix<Tm>(1);
         // loop from the rightmost site of MPS to the left
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
            assert(i == node.lindex);
            int tp = node.type;
            if(tp == 0 || tp == 1){
               // site on backbone with physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               int na_i = state[2*i], nb_i = state[2*i+1];
               qsym qc(3, na_i+nb_i, na_i+nb_i==1);
               if(debug) std::cout << "isite=" << i << " qc=" << qc << std::endl;
               // select allowed qleft
               const auto& qrow = site.info.qrow;
               qbond qleft;
               std::vector<linalg::matrix<Tm>> bmats2;
               for(int l=0; l<qrow.size(); l++){
                  auto ql = qrow.get_sym(l);
                  int dl = qrow.get_dim(l);
                  linalg::matrix<Tm> bmat(dl,1);
                  bool ifexist = false;
                  for(int r=0; r<qright.size(); r++){
                     auto qr = qright.get_sym(r);
                     int dr = qright.get_dim(r);
                     auto blk3 = site.get_rcf_symblk(ql,qr,qc);
                     if(blk3.empty()) continue;
                     // perform contraction Bnew[I] = \sum_J A[I,J]*B[J]
                     ifexist = true;
                     linalg::matrix<Tm> tmp(dl,dr,blk3.data());
                     auto tmp2 = linalg::xgemm("N","N",tmp,bmats[r]);
                     assert(tmp2.size() == bmat.size());
                     // <s[i]m[i]S[i-1]M[i-1]|S[i]M[i]> [consistent with csf.cpp] 
                     Tm cg = fock::cgcoeff(qc.ts(),qr.ts(),ql.ts(),na_i-nb_i,tmarray[i+1],tmarray[i]); 
                     linalg::xaxpy(tmp2.size(), cg, tmp2.data(), bmat.data());
                  }
                  if(ifexist){
                     if(debug) std::cout << " ql=" << ql << std::endl;
                     qleft.dims.push_back(std::make_pair(ql,dl));
                     bmats2.push_back(bmat);
                  }
               } // l
               if(debug) std::cout << " qleft.size=" << qleft.size() << std::endl;
               // in case this CTNS does not encode this csf, no such block 
               if(qleft.size() == 0) return coeff;
               qright = std::move(qleft);
               bmats = std::move(bmats2);
            } // tp
         } // i
         auto wf2 = icomb.get_wf2();
         if(debug){
            std::cout << "final contraction with rwfun" << std::endl;
            qright.print("qright");
            for(int i=0; i<bmats.size(); i++){
               bmats[i].print("bmat"+std::to_string(i));
            }
            wf2.print("wf2");
         }
         for(int r=0; r<qright.size(); r++){
            const auto& bmat = bmats[r];
            const auto& qr = qright.get_sym(r);
            int bc = wf2.info.qcol.existQ(qr);
            assert(bc != -1);
            auto wf2blk = wf2(0,bc);
            if(wf2blk.empty()) continue; // it is possible: ql={(1,0)}, qr={(0,0),{1,0}}
            auto wfcoeff = linalg::xgemm("N","N",wf2blk,bmat);
            assert(wfcoeff.rows() == n && wfcoeff.cols() == 1);
            linalg::xaxpy(n, 1.0, wfcoeff.data(), coeff.data());
         }
         if(debug) tools::print_vector(coeff,"coeff");
         return coeff;
      }

   // <csf|MPS>
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rcanon_detcoeff(const comb<Qm,Tm>& icomb,
            const std::vector<std::string>& detlist){
         std::cout << "\nctns::rcanon_detcoeff: size(dets)=" << detlist.size() << std::endl;
         for(int i=0; i<detlist.size(); i++){
            fock::onstate state(detlist[i]);
            auto coeff = rcanon_CIcoeff(icomb, state);
            std::cout << "det=" << state << std::endl;
            tools::print_vector(coeff, " coeffs", 8);
         }
      }
   template <typename Qm, typename Tm, std::enable_if_t<!Qm::ifabelian,int> = 0>
      void rcanon_csfcoeff(const comb<Qm,Tm>& icomb,
            const std::vector<std::string>& csflist){
         std::cout << "\nctns::rcanon_csfcoeff: size(csfs)=" << csflist.size() << std::endl;
         for(int i=0; i<csflist.size(); i++){
            fock::csfstate state(csflist[i]);
            auto coeff = rcanon_CIcoeff(icomb, state);
            std::cout << "csf=" << state << std::endl;
            tools::print_vector(coeff, " coeffs", 8);
         }
      }

} // ctns

#endif
