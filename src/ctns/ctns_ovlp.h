#ifndef CTNS_OVLP_H
#define CTNS_OVLP_H

#include "../core/onspace.h"
#include "ctns_comb.h"

namespace ctns{

/*
 Algorithms for CTNS:
 0. Check right canonical form
 1. <CTNS[i]|CTNS[j]> 
 2. <n|CTNS> and <CI|CTNS>
 3. random sampling from distribution p(n)=|<n|CTNS>|^2 and Sdiag
*/

// Check right canonical form
template <typename Km>
void rcanon_check(const comb<Km>& icomb,
		  const double thresh_ortho,
		  const bool ifortho){
   std::cout << "\nctns::rcanon_check thresh_ortho=" << thresh_ortho << std::endl;
   // loop over all sites
   int ntotal = icomb.topo.rcoord.size();
   for(int idx=0; idx<ntotal; idx++){
      auto p = icomb.topo.rcoord[idx];
      // check right canonical form -> A*[l'cr]A[lcr] = w[l'l] = Id
      auto qt2 = contract_qt3_qt3_cr(icomb.rsites.at(p),icomb.rsites.at(p));
      //qt2.to_matrix().print("qt2");
      double maxdiff = qt2.check_identityMatrix(thresh_ortho, false);
      int Dtot = qt2.qrow.get_dimAll();
      std::cout << "idx=" << idx << " node=" << p << " Dtot=" << Dtot 
		<< " maxdiff=" << std::scientific << maxdiff << std::endl;
      if((ifortho || (!ifortho && idx != ntotal-1)) && (maxdiff>thresh_ortho)){
	 std::cout << "error: deviate from identity matrix!" << std::endl;
         exit(1);
      }
   } // idx
}

// <CTNS[i]|CTNS[j]>: compute by a typical loop for right canonical form 
template <typename Km>
linalg::matrix<typename Km::dtype> get_Smat(const comb<Km>& icomb){ 
   std::cout << "\nctns::get_Smat" << std::endl;
   // loop over sites on backbone
   const auto& nodes = icomb.topo.nodes;
   qtensor2<typename Km::dtype> qt2_r, qt2_u;
   for(int i=icomb.topo.nbackbone-1; i>=0; i--){
      const auto& node = nodes[i][0];
      int tp = node.type;
      if(tp == 0 || tp == 1){
	 auto site = icomb.rsites.at(std::make_pair(i,0));
	 // merge wfuns[j,l] S[n,l,r] => tmp[n,j,r]
	 if(i == 0) site = contract_qt3_qt2_l(site, icomb.rwfuns); 
	 if(i == icomb.topo.nbackbone-1){
	    qt2_r = contract_qt3_qt3_cr(site,site);
	 }else{
	    auto qtmp = contract_qt3_qt2_r(site,qt2_r);
	    qt2_r = contract_qt3_qt3_cr(site,qtmp);
	 }
      }else if(tp == 3){
         for(int j=nodes[i].size()-1; j>=1; j--){
	    const auto& site = icomb.rsites.at(std::make_pair(i,j));
            if(j == nodes[i].size()-1){
	       qt2_u = contract_qt3_qt3_cr(site,site);
	    }else{
	       auto qtmp = contract_qt3_qt2_r(site,qt2_u);
	       qt2_u = contract_qt3_qt3_cr(site,qtmp);
	    }
	 } // j
	 // internal site without physical index
	 const auto& site = icomb.rsites.at(std::make_pair(i,0));
	 auto qtmp = contract_qt3_qt2_r(site,qt2_r); // ket
	 qtmp = contract_qt3_qt2_c(qtmp,qt2_u); // upper branch
	 qt2_r = contract_qt3_qt3_cr(site,qtmp); // bra
      }
   } // i
   auto Smat = qt2_r.to_matrix();
   return Smat;
}

// <n|CTNS[i]> by contracting the CTNS
template <typename Km>
std::vector<typename Km::dtype> rcanon_CIcoeff(const comb<Km>& icomb,
			       		       const fock::onstate& state){
   int n = icomb.get_nstate(); 
   std::vector<typename Km::dtype> coeff(n,0.0);
   // compute <n|CTNS> by contracting all sites
   const auto& nodes = icomb.topo.nodes;
   qtensor2<typename Km::dtype> qt2_r, qt2_u;
   for(int i=icomb.topo.nbackbone-1; i>=0; i--){
      const auto& node = nodes[i][0];
      int tp = node.type;
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
	 const auto& site = icomb.rsites.at(std::make_pair(i,0));
	 auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, node.pindex) );
	 if(i == icomb.topo.nbackbone-1){
	    qt2_r = qt2;
         }else{
	    qt2_r = qt2.dot(qt2_r); // (out,x)*(x,in)->(out,in)
	 }
      }else if(tp == 3){
	 // propogate symmetry from leaves down to backbone
         for(int j=nodes[i].size()-1; j>=1; j--){
	    const auto& site = icomb.rsites.at(std::make_pair(i,j));		 
	    const auto& nodej = nodes[i][j];
	    auto qt2 = site.fix_mid( occ2mdx(Km::isym, state, nodej.pindex) );
	    if(j == nodes[i].size()-1){
	       qt2_u = qt2;
	    }else{
	       qt2_u = qt2.dot(qt2_u);
	    }
         } // j
	 // internal site without physical index
	 const auto& site = icomb.rsites.at(std::make_pair(i,0));
	 qt2_u = qt2_u.T(); // permute row and col for contract_qt3_qt2_c
	 auto qt3 = contract_qt3_qt2_c(site,qt2_u); // contract upper sites
	 auto qt2 = qt3.fix_mid(std::make_pair(0,0));
	 qt2_r = qt2.dot(qt2_r); // contract right matrix
      }
   } // i
   auto wfcoeff = icomb.rwfuns.dot(qt2_r);
   assert(wfcoeff.rows() == 1 && wfcoeff.cols() == 1);
   const auto& blk = wfcoeff(0,0);
   if(blk.size() == 0) return coeff; // in case this CTNS does not encode this det 
   // compute fermionic sign changes to match ordering of orbitals
   double sgn = state.permute_sgn(icomb.topo.image2);
   std::transform(blk.col(0), blk.col(0)+n, coeff.begin(),
		  [sgn](const typename Km::dtype& x){ return sgn*x; });
   return coeff;
}

// ovlp[i,j] = <SCI[i]|CTNS[j]>
template <typename Km>
linalg::matrix<typename Km::dtype> rcanon_CIovlp(const comb<Km>& icomb,
				 		 const fock::onspace& space,
	                         		 const std::vector<std::vector<typename Km::dtype>>& vs){
   std::cout << "\nctns::rcanon_CIovlp" << std::endl;
   int n = icomb.get_nstate(); 
   int dim = space.size();
   // cmat[j,i] = <D[i]|CTNS[j]>
   linalg::matrix<typename Km::dtype> cmat(n,dim);
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_CIcoeff(icomb, space[i]);
      std::copy(coeff.begin(),coeff.end(),cmat.col(i));
   };
   // ovlp[i,j] = vs*[k,i] cmat[j,k] = (cmat[j,k] vs*[k,i])^T
   linalg::matrix<typename Km::dtype> vmat(vs);
   auto ovlp = linalg::xgemm("N","N",cmat,vmat.conj());
   return ovlp.T();
}

/*
// sampling of CTNS state to get {|det>,p(det)=|<det|Psi[i]>|^2}
template <typename Tm>
std::pair<fock::onstate,double> rcanon_random(const comb<Tm>& icomb, 
					      const int istate){
   const bool debug = false;
   fock::onstate state(2*icomb.get_nphysical());
   auto wf = icomb.get_state(istate); // initialize wf
   const auto& nodes = icomb.topo.nodes; 
   for(int i=0; i<icomb.topo.nbackbone; i++){
      int tp = nodes[i][0].type;
      if(tp == 0 || tp == 1){
	 const auto& site = icomb.rsites.at(std::make_pair(i,0));
	 auto qt3 = contract_qt3_qt2_l(site,wf);
	 // compute probability for physical index
         std::vector<qtensor2<Tm>> qt2n(4);
	 std::vector<double> weights(4);
	 for(int idx=0; idx<4; idx++){
            qt2n[idx] = qt3.fix_mid(get_mdx<Tm>(idx));
	    auto psi2 = qt2n[idx].dot(qt2n[idx].H()); // \sum_a |psi[n,a]|^2
	    weights[idx] = std::real(psi2(0,0)(0,0));
	 }
	 std::discrete_distribution<> dist(weights.begin(),weights.end());
	 int idx = dist(tools::generator);
	 assign_occupation_phys(state, nodes[i][0].pindex, idx);
	 wf = qt2n[idx];
      }else if(tp == 3){
	 const auto& site = icomb.rsites.at(std::make_pair(i,0));
	 auto qt3 = contract_qt3_qt2_l(site,wf);
	 // propogate upwards
	 for(int j=1; j<nodes[i].size(); j++){
	    const auto& sitej = icomb.rsites.at(std::make_pair(i,j));
	    // compute probability for physical index
            std::vector<qtensor3<Tm>> qt3n(4);
	    std::vector<double> weights(4);
	    for(int idx=0; idx<4; idx++){
	       auto qt2 = sitej.fix_mid(get_mdx<Tm>(idx));
	       qt3n[idx] = contract_qt3_qt2_c(qt3,qt2.T()); // purely change direction
               auto psi2 = contract_qt3_qt3_cr(qt3n[idx],qt3n[idx]); // \sum_ab |psi[n,a,b]|^2
	       weights[idx] = std::real(psi2(0,0)(0,0));
	    }
	    std::discrete_distribution<> dist(weights.begin(),weights.end());
	    int idx = dist(tools::generator);
	    assign_occupation_phys(state, nodes[i][j].pindex, idx);
	    qt3 = qt3n[idx];
         } // j
         wf = qt3.fix_mid(std::make_pair(0,0));
      }
   }
   // finally wf should be the corresponding CI coefficients: coeff0*sgn = coeff1
   auto coeff0 = wf(0,0)(0,0);
   if(debug){
      double sgn = state.permute_sgn(icomb.topo.image2); // from orbital ordering
      auto coeff1 = rcanon_CIcoeff(icomb, state)[istate];
      std::cout << " state=" << state 
                << " coeff0,sgn=" << coeff0 << "," << sgn
        	<< " coeff1=" << coeff1 << std::endl;
      assert(std::abs(coeff0*sgn-coeff1)<1.e-10);
   }
   double prob = std::norm(coeff0);
   return std::make_pair(state,prob);
}

// compute diagonal entropy via sampling 
template <typename Tm>
double rcanon_Sdiag_sample(const comb<Tm>& icomb,
		           const int istate,
		           const int nsample,  
		           const int nprt=10){ // no. of states to be printed
   const double cutoff = 1.e-12;
   std::cout << "\nctns::rcanon_Sdiag_sample istate=" << istate 
	     << " nsample=" << nsample << std::endl;
   auto t0 = tools::get_time();
   int noff = nsample/10;
   double Sd = 0.0, Sd2 = 0.0;
   std::map<fock::onstate,int> pop;
   for(int i=0; i<nsample; i++){
      auto pr = rcanon_random(icomb,istate);
      auto state = pr.first;
      auto pi = pr.second;
      // statistical analysis
      pop[state] += 1;
      double s = (pi < cutoff)? 0.0 : -log2(pi);
      double fac = 1.0/(i+1.0);
      Sd = (Sd*i + s)*fac;
      Sd2 = (Sd2*i + s*s)*fac;
      if((i+1)%noff == 0){
	 double std = sqrt((Sd2-Sd*Sd)/(i+1.e-10));
         auto t1 = tools::get_time();
	 double dt = tools::get_duration(t1-t0);
	 std::cout << " i=" << i << " Sd=" << Sd << " std=" << std
	           << " timing=" << dt << " s" << std::endl;	      
         t0 = tools::get_time();
      }
   }
   if(nprt > 0){
      int size = pop.size();
      std::cout << "sampled important determinants: size = " << size << std::endl;
      std::vector<fock::onstate> states(size);
      std::vector<int> counts(size);
      int i = 0;
      for(const auto& pr : pop){
	 states[i] = pr.first;
	 counts[i] = pr.second;
	 i++;
      }
      auto indx = tools::sort_index(counts,1);
      // compare the first n important dets by counts
      for(int i=0; i<std::min(size,nprt); i++){
	 int idx = indx[i];
	 fock::onstate state = states[idx];
	 auto ci = rcanon_CIcoeff(icomb, state)[istate];
	 std::cout << " i=" << i << " " << state
	           << " counts=" << counts[idx] 
	           << " p_i(sample)=" << counts[idx]/(1.0*nsample)
	           << " p_i(exact)=" << std::norm(ci) << std::endl;
      }
   }
   return Sd;
}

// exact computation of Sdiag, only for small system
template <typename Tm>
double rcanon_Sdiag_exact(const comb<Tm>& icomb,
			  const int istate){
   std::cout << "\nctns::rcanon_Sdiag_exact istate=" << istate;
   // setup FCI space
   qsym sym_state = icomb.get_sym_state();
   int ne = sym_state.ne(); // na+nb
   int tm = sym_state.tm(); // na-nb
   int ks = icomb.get_nphysical();
   const bool Htype = tools::is_complex<Tm>();
   fock::onspace fci_space;
   if(Htype){
      fci_space = fock::get_fci_space(2*ks,ne);
   }else{
      int na = (ne+tm)/2, nb = ne - na;
      fci_space = fock::get_fci_space(ks,na,nb); 
   }
   int dim = fci_space.size();
   std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI=" << dim << std::endl;
   // compute exact coefficients <n|CTNS>
   std::vector<Tm> coeff(dim,0.0);
   for(int i=0; i<dim; i++){
      const auto& state = fci_space[i];
      coeff[i] = rcanon_CIcoeff(icomb, state)[istate];
      std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
   }
   double Sdiag = fock::coeff_entropy(coeff);
   std::cout << "Sdiag(exact) = " << Sdiag << std::endl;
   return Sdiag;
}
*/

} // ctns

#endif
