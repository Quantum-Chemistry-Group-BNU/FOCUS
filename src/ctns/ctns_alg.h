#ifndef CTNS_ALG_H
#define CTNS_ALG_H

/*

 Algorithms for CTNS:

  0. rcanon_check: check rsites in right canonical form (RCF)
  1. get_Smat: <CTNS[i]|CTNS[j]> 
  2. rcanon_CIcoeff: <n|CTNS>
     rcanon_CIovlp: <CI|CTNS>
     rcanon_CIcoeff_check
  3. rcanon_Sdiag_exact:
     rcanon_random: random sampling from distribution p(n)=|<n|CTNS>|^2
     rcanon_Sdiag_sample: compute Sdiag via sampling

*/

#include "alg_ova.h"

namespace ctns{

// Algorithm 0:
// Check right canonical form
template <typename Km>
void rcanon_check(const comb<Km>& icomb,
		  const double thresh_ortho,
		  const bool ifortho=true){
   std::cout << "\nctns::rcanon_check thresh_ortho=" 
	     << std::scientific << std::setprecision(2) << thresh_ortho 
	     << std::endl;
   // loop over all sites
   for(int idx=0; idx<icomb.topo.ntotal; idx++){
      auto p = icomb.topo.rcoord[idx];
      // check right canonical form -> A*[l'cr]A[lcr] = w[l'l] = Id
      auto qt2 = contract_qt3_qt3_cr(icomb.rsites[idx],icomb.rsites[idx]);
      double maxdiff = qt2.check_identityMatrix(thresh_ortho, false);
      int Dtot = qt2.row_dimAll();
      std::cout << " idx=" << idx << " node=" << p << " Dtot=" << Dtot 
		<< " maxdiff=" << std::scientific << maxdiff << std::endl;
      if((ifortho || (!ifortho && idx != icomb.topo.ntotal-1)) && (maxdiff>thresh_ortho)){
	 tools::exit("error: deviate from identity matrix!");
      }
   } // idx
}

// Algorithm 1:
// <CTNS[i]|CTNS[j]>: compute by a typical loop for right canonical form 

// Algorithm 2:
// <n|CTNS[i]> by contracting the CTNS

// check rcanon_CIcoeff
template <typename Km>
int rcanon_CIcoeff_check(const comb<Km>& icomb,
		         const fock::onspace& space,
	                 const std::vector<std::vector<typename Km::dtype>>& vs,
			 const double thresh=1.e-8){
   std::cout << "\nctns::rcanon_CIcoeff_check" << std::endl;
   int n = icomb.get_nroots(); 
   int dim = space.size();
   double maxdiff = -1.e10;
   // cmat[j,i] = <D[i]|CTNS[j]>
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_CIcoeff(icomb, space[i]);
      std::cout << " i=" << i << " state=" << space[i] << std::endl;
      for(int j=0; j<n; j++){
	 auto diff = std::abs(coeff[j] - vs[j][i]);
         std::cout << "   j=" << j << " <n|CTNS[j]>=" << coeff[j] 
		   << " <n|CI[j]>=" << vs[j][i] 
		   << " diff=" << diff << std::endl;
	 maxdiff = std::max(maxdiff, diff);
      }
   }
   std::cout << "maxdiff = " << maxdiff << " thresh=" << thresh << std::endl;
   if(maxdiff > thresh) tools::exit("error: too large maxdiff in rcanon_CIcoeff_check!");
   return 0;
}

// ovlp[i,n] = <SCI[i]|CTNS[n]>
template <typename Km>
linalg::matrix<typename Km::dtype> rcanon_CIovlp(const comb<Km>& icomb,
				 		 const fock::onspace& space,
	                         		 const std::vector<std::vector<typename Km::dtype>>& vs){
   using Tm = typename Km::dtype;
   std::cout << "\nctns::rcanon_CIovlp" << std::endl;
   int n = icomb.get_nroots(); 
   int dim = space.size();
   // cmat[n,i] = <D[i]|CTNS[n]>
   linalg::matrix<Tm> cmat(n,dim);
   for(int i=0; i<dim; i++){
      auto coeff = rcanon_CIcoeff(icomb, space[i]);
      linalg::xcopy(n, coeff.data(), cmat.col(i));
   }
   // ovlp[i,n] = vs*[k,i] cmat[n,k]
   linalg::matrix<Tm> vmat(vs);
   auto ovlp = linalg::xgemm("C","T",vmat,cmat);
   return ovlp;
}

// Algorithm 3:
// exact computation of Sdiag, only for small system
template <typename Km>
double rcanon_Sdiag_exact(const comb<Km>& icomb,
			  const int iroot,
   		          const double thresh_print=1.e-10){
   using Tm = typename Km::dtype; 
   std::cout << "\nctns::rcanon_Sdiag_exact iroot=" << iroot
             << " thresh_print=" << thresh_print << std::endl;
   // setup FCI space
   qsym sym_state = icomb.get_sym_state();
   int ne = sym_state.ne(); 
   int ks = icomb.get_nphysical();
   fock::onspace fci_space;
   if(Km::isym == 0){
      fci_space = fock::get_fci_space(2*ks);
   }else if(Km::isym == 1){
      fci_space = fock::get_fci_space(2*ks,ne);
   }else if(Km::isym == 2){
      int tm = sym_state.tm(); 
      int na = (ne+tm)/2, nb = ne-na;
      fci_space = fock::get_fci_space(ks,na,nb); 
   }
   int dim = fci_space.size();
   std::cout << " ks=" << ks << " sym=" << sym_state << " dimFCI=" << dim << std::endl;
   
   // brute-force computation of exact coefficients <n|CTNS>
   std::vector<Tm> coeff(dim,0.0);
   for(int i=0; i<dim; i++){
      const auto& state = fci_space[i];
      coeff[i] = rcanon_CIcoeff(icomb, state)[iroot];
      if(std::abs(coeff[i]) < thresh_print) continue;
      std::cout << " i=" << i << " " << state << " coeff=" << coeff[i] << std::endl; 
   }
   double Sdiag = fock::coeff_entropy(coeff);
   double ovlp = std::pow(linalg::xnrm2(dim,&coeff[0]),2); 
   std::cout << "ovlp=" << ovlp << " Sdiag(exact)=" << Sdiag << std::endl;
   
   // check: computation by sampling CI vector
   std::vector<double> weights(dim,0.0);
   std::transform(coeff.begin(), coeff.end(), weights.begin(),
		  [](const Tm& x){ return std::norm(x); });
   std::discrete_distribution<> dist(weights.begin(),weights.end());
   const int nsample = 1e6;
   int noff = nsample/10;
   const double cutoff = 1.e-12;
   double Sd = 0.0, Sd2 = 0.0, std = 0.0;
   for(int i=0; i<nsample; i++){
      int idx = dist(tools::generator);
      auto ci2 = weights[idx];
      double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
      double fac = 1.0/(i+1.0);
      Sd = (Sd*i + s)*fac;
      Sd2 = (Sd2*i + s*s)*fac;
      if((i+1)%noff == 0){
         std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
	 std::cout << " i=" << i << " Sd=" << Sd << " std=" << std 
		   << " range=(" << Sd-std << "," << Sd+std << ")"
		   << std::endl;
      }
   }
   return Sdiag;
}

// compute diagonal entropy via sampling:
// S = -p[i]log2p[i] = - (sum_i p[i]) <log2p[i] > = -<psi|psi>*<log2p[i]>
template <typename Km>
double rcanon_Sdiag_sample(const comb<Km>& icomb,
		           const int iroot,
		           const int nsample,  
		           const int nprt=10){ // no. of largest states to be printed
   const double cutoff = 1.e-12;
   std::cout << "\nctns::rcanon_Sdiag_sample iroot=" << iroot 
	     << " nsample=" << nsample 
	     << " nprt=" << nprt << std::endl;
   auto t0 = tools::get_time();
   const int noff = nsample/10;
   // In case CTNS is not normalized 
   double ovlp = std::abs(get_Smat(icomb)(iroot,iroot));
   std::cout << "<CTNS[i]|CTNS[i]> = " << ovlp << std::endl; 
   // start sampling
   double Sd = 0.0, Sd2 = 0.0, std = 0.0;
   std::map<fock::onstate,int> pop;
   for(int i=0; i<nsample; i++){
      auto pr = rcanon_random(icomb,iroot);
      auto state = pr.first;
      auto ci2 = pr.second;
      // statistical analysis
      pop[state] += 1;
      double s = (ci2 < cutoff)? 0.0 : -log2(ci2)*ovlp;
      double fac = 1.0/(i+1.0);
      Sd = (Sd*i + s)*fac;
      Sd2 = (Sd2*i + s*s)*fac;
      if((i+1)%noff == 0){
         std = std::sqrt((Sd2-Sd*Sd)/(i+1.e-10));
         auto t1 = tools::get_time();
	 double dt = tools::get_duration(t1-t0);
	 std::cout << " i=" << i << " Sd=" << Sd << " std=" << std
	           << " timing=" << dt << " s" << std::endl;	      
         t0 = tools::get_time();
      }
   }
   // print important determinants
   if(nprt > 0){
      int size = pop.size();
      std::cout << "sampled important determinants: pop.size=" << size << std::endl; 
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
      int sum = 0;
      for(int i=0; i<std::min(size,nprt); i++){
	 int idx = indx[i];
	 fock::onstate state = states[idx];
	 auto ci = rcanon_CIcoeff(icomb, state)[iroot];
	 sum += counts[idx];
	 std::cout << " i=" << i << " " << state
	           << " counts=" << counts[idx] 
	           << " p_i(sample)=" << counts[idx]/(1.0*nsample)
	           << " p_i(exact)=" << std::norm(ci)/ovlp 
		   << " c_i(exact)=" << ci/std::sqrt(ovlp)
		   << std::endl;
      }
      std::cout << "accumulated counts=" << sum 
	        << " nsample=" << nsample 
		<< " per=" << 1.0*sum/nsample << std::endl;
   }
   return Sd;
}

} // ctns

#endif
