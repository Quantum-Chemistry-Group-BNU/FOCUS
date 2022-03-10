#ifndef ALG_OVA_H
#define ALG_OVA_H

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"

namespace ctns{

// <CTNS[i]|CTNS[j]>: compute by a typical loop for right canonical form 
template <typename Km>
linalg::matrix<typename Km::dtype> get_Smat(const comb<Km>& icomb){ 
   // loop over sites on backbone
   const auto& nodes = icomb.topo.nodes;
   const auto& rindex = icomb.topo.rindex;
   stensor2<typename Km::dtype> qt2_r, qt2_u;
   for(int i=icomb.topo.nbackbone-1; i>0; i--){
      const auto& node = nodes[i][0];
      int tp = node.type;
      if(tp == 0 || tp == 1){
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
	 if(i == icomb.topo.nbackbone-1){
	    qt2_r = contract_qt3_qt3_cr(site,site);
	 }else{
	    auto qtmp = contract_qt3_qt2_r(site,qt2_r);
	    qt2_r = contract_qt3_qt3_cr(site,qtmp);
	 }
      }else if(tp == 3){
         for(int j=nodes[i].size()-1; j>=1; j--){
	    const auto& site = icomb.rsites[rindex.at(std::make_pair(i,j))];
            if(j == nodes[i].size()-1){
	       qt2_u = contract_qt3_qt3_cr(site,site);
	    }else{
	       auto qtmp = contract_qt3_qt2_r(site,qt2_u);
	       qt2_u = contract_qt3_qt3_cr(site,qtmp);
	    }
	 } // j
	 // internal site without physical index
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
	 auto qtmp = contract_qt3_qt2_r(site,qt2_r); // ket
	 qtmp = contract_qt3_qt2_c(qtmp,qt2_u); // upper branch
	 qt2_r = contract_qt3_qt3_cr(site,qtmp); // bra
      }
   } // i
   // first merge: sum_l rwfuns[j,l]*site0[l,r,n] => site[j,r,n]
   const auto& site0 = icomb.rsites[rindex.at(std::make_pair(0,0))];
   site0.print("site0");
   icomb.rwfuns.print("rwfuns");
   auto site = contract_qt3_qt2_l(site0,icomb.rwfuns); 
   auto qtmp = contract_qt3_qt2_r(site,qt2_r);
   qt2_r = contract_qt3_qt3_cr(site,qtmp);
   auto Smat = qt2_r.to_matrix();
   return Smat;
}

// <n|CTNS[i]> by contracting the CTNS
template <typename Km>
std::vector<typename Km::dtype> rcanon_CIcoeff(const comb<Km>& icomb,
			       		       const fock::onstate& state){
   using Tm = typename Km::dtype;
   // compute <n|CTNS> by contracting all sites
   const auto& nodes = icomb.topo.nodes;
   const auto& rindex = icomb.topo.rindex;
   stensor2<Tm> qt2_r, qt2_u;
   for(int i=icomb.topo.nbackbone-1; i>=0; i--){
      const auto& node = nodes[i][0];
      int tp = node.type;
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
         // given occ pattern, extract the corresponding qblock
	 auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, node.pindex) ); 
	 if(i == icomb.topo.nbackbone-1){
	    qt2_r = std::move(qt2);
         }else{
	    qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
	 }
      }else if(tp == 3){
	 // propogate symmetry from leaves down to backbone
         for(int j=nodes[i].size()-1; j>=1; j--){
	    const auto& site = icomb.rsites[rindex.at(std::make_pair(i,j))];
	    const auto& nodej = nodes[i][j];
	    auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, nodej.pindex) );
	    if(j == nodes[i].size()-1){
	       qt2_u = std::move(qt2);
	    }else{
	       qt2_u = qt2.dot(qt2_u);
	    }
         } // j
	 // internal site without physical index
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
	 // contract upper matrix: permute row and col for contract_qt3_qt2_c
	 auto qt3 = contract_qt3_qt2_c(site,qt2_u.T());
	 auto qt2 = qt3.fix_mid( std::make_pair(0,0) );
	 qt2_r = qt2.dot(qt2_r); // contract right matrix
      } // tp
   } // i
   auto wfcoeff = icomb.rwfuns.dot(qt2_r);
   assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
   // finally return coeff = <n|CTNS> 
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

// Sampling CTNS to get {|det>,p(det)=|<det|Psi[i]>|^2} 
// In case that CTNS is unnormalized, p(det) is also unnormalized. 
template <typename Km>
std::pair<fock::onstate,double> rcanon_random(const comb<Km>& icomb, 
					      const int iroot,
   					      const bool debug=false){
   if(debug) std::cout << "\nctns::rcanon_random iroot=" << iroot << std::endl; 
   using Tm = typename Km::dtype; 
   fock::onstate state(2*icomb.get_nphysical());
   auto wf = icomb.get_iroot(iroot); // initialize wf
   const auto& nodes = icomb.topo.nodes; 
   const auto& rindex = icomb.topo.rindex;
   for(int i=0; i<icomb.topo.nbackbone; i++){
      int tp = nodes[i][0].type;
      if(tp == 0 || tp == 1){
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
	 auto qt3 = contract_qt3_qt2_l(site,wf);
	 // compute probability for physical index
         std::vector<stensor2<Tm>> qt2n(4);
	 std::vector<double> weights(4);
	 for(int idx=0; idx<4; idx++){
	    qt2n[idx] = qt3.fix_mid( idx2mdx(Km::isym, idx) );
	    // \sum_a |psi[n,a]|^2
	    auto psi2 = qt2n[idx].dot(qt2n[idx].H()); 
	    weights[idx] = std::real(psi2(0,0)(0,0));
	 }
	 std::discrete_distribution<> dist(weights.begin(),weights.end());
	 int idx = dist(tools::generator);
	 idx2occ(state, nodes[i][0].pindex, idx);
	 wf = std::move(qt2n[idx]);
      }else if(tp == 3){
	 const auto& site = icomb.rsites[rindex.at(std::make_pair(i,0))];
	 auto qt3 = contract_qt3_qt2_l(site,wf);
	 // propogate upwards
	 for(int j=1; j<nodes[i].size(); j++){
	    const auto& sitej = icomb.rsites[rindex.at(std::make_pair(i,j))];
	    // compute probability for physical index
            std::vector<stensor3<Tm>> qt3n(4);
	    std::vector<double> weights(4);
	    for(int idx=0; idx<4; idx++){
	       auto qt2 = sitej.fix_mid( idx2mdx(Km::isym, idx) );
	       // purely change direction
	       qt3n[idx] = contract_qt3_qt2_c(qt3,qt2.T()); 
	       // \sum_ab |psi[n,a,b]|^2
               auto psi2 = contract_qt3_qt3_cr(qt3n[idx],qt3n[idx]); 
	       weights[idx] = std::real(psi2(0,0)(0,0));
	    }
	    std::discrete_distribution<> dist(weights.begin(),weights.end());
	    int idx = dist(tools::generator);
	    idx2occ(state, nodes[i][j].pindex, idx);
	    qt3 = std::move(qt3n[idx]);
         } // j
         wf = qt3.fix_mid(std::make_pair(0,0));
      }
   }
   // finally wf should be the corresponding CI coefficients: coeff0*sgn = coeff1
   auto coeff0 = wf(0,0)(0,0);
   if(debug){
      double sgn = state.permute_sgn(icomb.topo.image2); // from orbital ordering
      auto coeff1 = rcanon_CIcoeff(icomb, state)[iroot];
      std::cout << " state=" << state 
                << " coeff0,sgn=" << coeff0 << "," << sgn
        	<< " coeff1=" << coeff1 
		<< " diff=" << coeff0*sgn-coeff1 << std::endl;
      assert(std::abs(coeff0*sgn-coeff1)<1.e-10);
   }
   double prob = std::norm(coeff0);
   return std::make_pair(state,prob);
}

} // ctns

#endif
