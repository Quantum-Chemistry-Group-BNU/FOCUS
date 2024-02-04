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
#include "ctns_comb.h"

namespace ctns{

   // Algorithm 2:
   // <n|CTNS[i]> by contracting the CTNS
   template <typename Qm, typename Tm>
      std::vector<Tm> rcanon_CIcoeff(const comb<Qm,Tm>& icomb,
            const fock::onstate& state){
         std::cout << "X state=" << state << std::endl;
         // compute <n|CTNS> by contracting all sites
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         qtensor2<Qm::ifabelian,Tm> qt2_r, qt2_u;
         for(int i=icomb.topo.nbackbone-1; i>=0; i--){
            std::cout << "\ni=" << i << std::endl;
            const auto& node = nodes[i][0];
            int tp = node.type;
            if(tp == 0 || tp == 1){
               // site on backbone with physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               // given occ pattern, extract the corresponding qblock
               auto qt2 = site.fix_mid( occ2mdx(Qm::isym, state, node.pindex) ); 
               
               qt2.print("qt2");
               qt2_r.print("qt2_r.old");
               if(i == icomb.topo.nbackbone-1){
                  qt2_r = std::move(qt2);
               }else{
                  qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
               }
               qt2_r.print("qt2_r.new");

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
         std::cout << "wfcoeff" << std::endl;
         const auto& wfcoeff = icomb.get_wf2().dot(qt2_r);
         assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
         // finally return coeff = <n|CTNS[i]> as a vector 
         int n = icomb.get_nroots(); 
         std::vector<Tm> coeff(n,0.0);
         // in case this CTNS does not encode this det, no such block 
         const auto blk2 = wfcoeff(0,0);
         if(blk2.empty()) return coeff; 
         assert(blk2.size() == n);
         // compute fermionic sign changes to match ordering of orbitals
         double sgn = state.permute_sgn(icomb.topo.image2);
         linalg::xaxpy(n, sgn, blk2.data(), coeff.data());
         return coeff;
      }

   // check rcanon_CIcoeff
   template <typename Qm, typename Tm>
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
   template <typename Qm, typename Tm>
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
