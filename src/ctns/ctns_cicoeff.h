#ifndef CTNS_CICOEFF_H
#define CTNS_CICOEFF_H

/*
   Algorithms for CTNS:

   2. rcanon_CIcoeff: <n|CTNS>
      rcanon_CIovlp: <CI|CTNS>
      rcanon_CIcoeff_check
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "../core/csf.h"
#include "ctns_comb.h"

namespace ctns{

   // Algorithm 2:
   // <n|CTNS[i]> by contracting the CTNS
   //
   // |n>=|n1n2n3>, |q> can be |q1q3q2>, |psi>=|q>*psi(q)
   // then <n|psi> = <n|q>*psi(q), <n|q> is the sign
   //
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      std::vector<Tm> rcanon_CIcoeff(const comb<Qm,Tm>& icomb,
            const fock::onstate& state){
         // finally return coeff = <n|CTNS[i]> as a vector 
         int n = icomb.get_nroots(); 
         std::vector<Tm> coeff(n,0.0);
         // compute <n|CTNS> by contracting all sites
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         qtensor2<Qm::ifabelian,Tm> qt2_r, qt2_u;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
            int tp = node.type;
            if(tp == 0 || tp == 1){
               // site on backbone with physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               // given occ pattern, extract the corresponding qblock
               auto qt2 = site.fix_mid( occ2mdx(Qm::isym, state, node.pindex) ); 
               if(i == icomb.topo.nbackbone-1){
                  qt2_r = std::move(qt2);
               }else{
                  qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
               }
            }else if(tp == 3){
               // propogate symmetry from leaves down to backbone
               for(int j=nodes[i].size()-1; j>=1; j--){
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,j))];
                  const auto& nodej = nodes[i][j];
                  auto qt2 = site.fix_mid( occ2mdx(Qm::isym, state, nodej.pindex) );
                  if(j == nodes[i].size()-1){
                     qt2_u = std::move(qt2);
                  }else{
                     qt2_u = qt2.dot(qt2_u);
                  }
               } // j
               // internal site without physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               // contract upper matrix: permute row and col for contract_qt3_qt2_c
               auto qt3 = contract_qt3_qt2("c",site,qt2_u.P());
               auto qt2 = qt3.fix_mid( std::make_pair(0,0) );
               qt2_r = qt2.dot(qt2_r); // contract right matrix
            } // tp
         } // i
         const auto& wfcoeff = icomb.get_wf2().dot(qt2_r);
         assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
         // in case this CTNS does not encode this det, no such block 
         const auto blk2 = wfcoeff(0,0);
         if(blk2.empty()) return coeff; 
         assert(blk2.size() == n);
         // compute fermionic sign changes to match ordering of orbitals
         double sgn = state.permute_sgn(icomb.topo.image2);
         linalg::xaxpy(n, sgn, blk2.data(), coeff.data());
         return coeff;
      }

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
         std::cout << "csf=" << state << std::endl;
         tools::print_vector(narray,"narray");
         tools::print_vector(tsarray,"tsarray");
         //
         // csf=222000 [from right to left]
         //  narray= 6 6 6 6 4 2 0
         //  tsarray= 0 0 0 0 0 0 0
         */
         // compute <n|CTNS> by contracting all sites
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         linalg::matrix<Tm> bmat;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            const auto& node = nodes[i][0];
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

   // check rcanon_CIcoeff
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      int rcanon_CIcoeff_check(const comb<Qm,Tm>& icomb,
            const fock::onspace& space,
            const linalg::matrix<Tm>& vs,
            const double thresh=1.e-8){
         std::cout << "\nctns::rcanon_CIcoeff_check" << std::endl;
         int n = icomb.get_nroots(); 
         size_t dim = space.size();
         double maxdiff = -1.e10;
         // cmat[j,i] = <D[i]|CTNS[j]>
         for(int i=0; i<dim; i++){
            auto coeff = rcanon_CIcoeff(icomb, space[i]);
            std::cout << " i=" << i << " state=" << space[i] << std::endl;
            for(int j=0; j<n; j++){
               auto diff = std::abs(coeff[j] - vs(i,j));
               std::cout << "   j=" << j << " <n|CTNS[j]>=" << coeff[j] 
                  << " <n|CI[j]>=" << vs(i,j)
                  << " diff=" << diff << std::endl;
               maxdiff = std::max(maxdiff, diff);
            }
         }
         std::cout << "maxdiff = " << maxdiff << " thresh=" << thresh << std::endl;
         if(maxdiff > thresh) tools::exit("error: too large maxdiff in rcanon_CIcoeff_check!");
         return 0;
      }

   // ovlp[i,n] = <SCI[i]|CTNS[n]>
   template <typename Qm, typename Tm, std::enable_if_t<Qm::ifabelian,int> = 0>
      linalg::matrix<Tm> rcanon_CIovlp(const comb<Qm,Tm>& icomb,
            const fock::onspace& space,
            const linalg::matrix<Tm>& vs){
         std::cout << "\nctns::rcanon_CIovlp" << std::endl;
         int n = icomb.get_nroots(); 
         size_t dim = space.size();
         // cmat[n,i] = <D[i]|CTNS[n]>
         linalg::matrix<Tm> cmat(n,dim);
         for(int i=0; i<dim; i++){
            auto coeff = rcanon_CIcoeff(icomb, space[i]);
            linalg::xcopy(n, coeff.data(), cmat.col(i));
         }
         // ovlp[i,n] = vs*[k,i] cmat[n,k]
         auto ovlp = linalg::xgemm("C","T",vs,cmat);
         return ovlp;
      }

} // ctns

#endif
