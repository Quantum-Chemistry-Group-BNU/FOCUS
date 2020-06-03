#include "../core/linalg.h"
#include "../core/analysis.h"
#include "../core/tools.h"
#include "tns_comb.h" 
#include "tns_qtensor.h"
#include "tns_oper.h"
#include <random>

using namespace std;
using namespace linalg;
using namespace tns;
using namespace fock;

// sampling of Comb state to get {|det>,p(det)=|<det|Psi[i]>|^2}
pair<onstate,double> comb::rcanon_sampling(const int istate){
   // selector
   qsym sym_state = rsites[make_pair(0,0)].qrow.begin()->first;  
   int nstate = rsites[make_pair(0,0)].qrow[sym_state]; 
   matrix vec_l(1,nstate);
   vec_l(0,istate) = 1.0; 
   // loop over sites on backbone
   auto q00 = make_pair(phys_sym[0],phys_sym[0]);
   auto q01 = make_pair(phys_sym[1],phys_sym[1]);
   auto q10 = make_pair(phys_sym[2],phys_sym[2]);
   auto q11 = make_pair(phys_sym[3],phys_sym[3]);
   onstate state(2*nphysical);
   qsym sym_l = sym_state, sym_p, sym_r;
   for(int i=0; i<nbackbone; i++){
      auto p = make_pair(i,0);
      int tp = type[p];
      if(tp == 0 || tp == 1){
         // site on backbone with physical index
	 auto qt2cr = contract_qt3_vec_l(rsites[p],sym_l,vec_l); // c<->r
	 auto qt2 = qt2cr.get_rdm_row();
	 vector<double> weights = {qt2.qblocks[q00](0,0),
	            		   qt2.qblocks[q01](0,0),
                    		   qt2.qblocks[q10](0,0),
                    		   qt2.qblocks[q11](0,0)};
	 std::discrete_distribution<> dist(weights.begin(),weights.end());
	 int np = dist(tools::generator);
   	 int k = topo[i][0];
	 if(np == 0){
	    state[2*k] = 0; state[2*k+1] = 0;
	 }else if(np == 1){
	    state[2*k] = 0; state[2*k+1] = 1;
	 }else if(np == 2){
	    state[2*k] = 1; state[2*k+1] = 0;
	 }else if(np == 3){
	    state[2*k] = 1; state[2*k+1] = 1;
	 }
	 // update
	 sym_p = phys_sym[np];
	 sym_r = sym_l - sym_p;
	 vec_l = qt2cr.qblocks[make_pair(sym_p,sym_r)];
	 sym_l = sym_r;
      }else if(tp == 3){
/*
	 auto qt2_cr = contract_qt3_lvec(rsites[p],sym_l,vec_l); 
	 // propogate upwards
	 for(int j=1; j<topo[i].size(); j++){
	    auto pu = make_pair(i,j);
	    auto qt3 = contract_qt3_qt2_l(rsites[pu],qt2_cr.P());
	    auto qt2 = contract_qt3_qt3_lr(qt3,qt3);
       	    vector<double> weights = {qt2.qblocks[q00](0,0),
         	            	      qt2.qblocks[q01](0,0),
                             	      qt2.qblocks[q10](0,0),
                             	      qt2.qblocks[q11](0,0)};
	    std::discrete_distribution<> dist(weights.begin(),weights.end());
	    int np = dist(tools::generator);
   	    int k = topo[i][0];
	    if(np == 0){
	       state[2*k] = 0; state[2*k+1] = 0;
	    }else if(np == 1){
	       state[2*k] = 0; state[2*k+1] = 1;
	    }else if(np == 2){
	       state[2*k] = 1; state[2*k+1] = 0;
	    }else if(np == 3){
	       state[2*k] = 1; state[2*k+1] = 1;
	    }
	    sym_p = phys_sym[np];
	    qt2_cr = qt3.get,sym_p);
         }
	 vec_l = qt2_cr.qblocks[];
	 sym_l = sym_r;

		 sym_r = sym_l - sym_p;
	 auto key = make_tuple(sym_p,sym_r);
	 // update
	 vec_l = qt2_cr.qblocks[key];
	 sym_l = sym_r;
*/
      }
   }
   // finally vec_l should be the corresponding CI coefficients
   double coeff0 = vec_l(0,0);
   double coeff1 = rcanon_CIcoeff(state)[istate];
   double prob = coeff0*coeff0;
   return make_pair(state,prob);
}

double comb::rcanon_sampling_Sd(const int nsample, const int istate){
   cout << "\ncomb::rcanon_sampling_Sd nsample=" << nsample << " istate=" << istate << endl;
   const double cutoff = 1.e-12;
   int nprt = nsample/10;
   double Sd = 0.0;
   double Sd2 = 0.0;
   for(int i=0; i<nsample; i++){
      auto pr = rcanon_sampling(istate);
      double pi = pr.second;
      double s = (pi < cutoff)? 0.0 : -log2(pi);
      double fac = 1.0/(i+1.0);
      Sd = (Sd*i + s)*fac;
      Sd2 = (Sd2*i + s*s)*fac;
      if((i+1)%nprt == 0){
	 double rn = sqrt((Sd2-Sd*Sd)/(i+1.e-10));
         cout << "i=" << i 
 	      << " Sd=" << Sd
	      << " rn=" << rn 
	      << endl;	      
      }
   }
   return Sd;
}

// check by explict list all dets in the FCI space
void comb::rcanon_sampling_check(const int istate){
   cout << "\ncomb::rcanon_sampling_check istate=" << istate;
   qsym sym_state = rsites[make_pair(0,0)].qrow.begin()->first;
   int ne = sym_state.ne();
   int na = sym_state.na();
   int ks = nphysical;
   onspace fci_space = get_fci_space(ks,na,ne-na);
   int dim = fci_space.size();
   cout << " ks/ne/na/dimFCI=" << ks << "," << ne << "," << na << "," << dim << endl;
   // exact values
   vector<double> coeff(dim,0.0);
   for(int i=0; i<dim; i++){
      const auto& det = fci_space[i];
      coeff[i] = rcanon_CIcoeff(det)[istate];
      cout << " i=" << i 
	   << " " << det.to_string2() 
	   << " coeff=" << coeff[i] 
	   << endl; 
   }
   double Sd = coeff_entropy(coeff);
   cout << "Sd(exact) = " << Sd << endl;
}
