#include "../settings/global.h"
#include "../core/onstate.h"
#include "../core/dvdson.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "fci.h"
#include "sci.h"

#include <bitset>

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;
using namespace sci;

heatbath_table::heatbath_table(const integral::two_body& int2e){
   cout << "\nheatbath_table::heatbath_table" << endl;
   auto t0 = global::get_time();
   bool debug = false;

   int k = int2e.sorb;
   sorb = k;
   for(int i=0; i<k; i++){
      for(int j=0; j<i; j++){
	 int ij = i*(i-1)/2+j;
	 for(int p=0; p<k; p++){
 	    for(int q=0; q<p; q++){
	       // <ij||pq>=[ip|jq]-[iq|jp] (i>j, p>q)
	       double mag = abs(int2e.get(i,p,j,q) - int2e.get(i,q,j,p));
	       if(mag > thresh){
	          eri[ij].insert(make_pair(mag,p*(p-1)/2+q));
	       }
	    } // q
	 } // p
      } // j
   } // i
   auto t1 = global::get_time();
   cout << "timing for heatbath_table::heatbath_table : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   
   if(debug){
      cout << defaultfloat << setprecision(12);
      for(int ij=0; ij<k*(k-1)/2; ij++){
         auto pr = tools::inverse_pair0(ij);
	 int i = pr.first, j = pr.second;
	 cout << "i,j=" << i << "," << j << " eri[ij] : " << eri[ij].size() << endl;
	 for(const auto& p : eri[ij]){
            auto rs = tools::inverse_pair0(p.second);
            cout << " val=" << p.first 
     		 << " r,s=" << rs.first << "," << rs.second << endl;
	 }
      }
   } // debug
}
     
// expand variational subspace
void sci::expand_varSpace(onspace& space, 
			  unordered_set<onstate>& varSpace,
	       	          const integral::two_body& int2e,
	       	          const integral::one_body& int1e,
		          const heatbath_table& hbtab, 
			  vector<double>& cmax, 
			  const double eps1){
   cout << "\nsci::expand_varSpace dim = " << space.size() << endl;
   auto t0 = global::get_time();
   bool debug = false; //true;

   onstate state = space[0];
   int no = state.nelec();
   int nv = state.size() - no;
   int nsingles = no*nv;
   int dim = space.size();
   int inew = 0;
   // loop over each det |Di> in V
   for(int idx=0; idx<dim; idx++){
      // select |Dj> if |<Dj|H|Di>cmax[i]|>eps1 && |Dj> is not in V 
      state = space[idx];
      if(debug){
	 cout << " i=" << idx << " " << state.to_string2() 
	      << " (N,Na,Nb)=" << state.nelec()
	      << "," << state.nelec_a() << "," << state.nelec_b()
	      << " cmax=" << cmax[idx] << endl;	 
      } 
      vector<int> olst,vlst;
      state.get_olst(olst);
      state.get_vlst(vlst);
/*
      // singles
      for(int ia=0; ia<nsingles; ia++){
         int i = ia%no, a = ia/no;
	 onstate state1(state);
	 state1[olst[i]] = 0;
	 state1[vlst[a]] = 1;
	 auto HijDiff = fock::get_HijS(state1,state,int2e,int1e);
	 if(abs(HijDiff.first)*cmax[idx] > eps1){
	    auto search = varSpace.find(state1);
	    if(search == varSpace.end()){
	       varSpace.insert(state1);
	       space.push_back(state1);
	       if(debug){
		  cout << "   " << inew 
	               << " S(i->a) = " << symbol(olst[i]) 
		       << "->" << symbol(vlst[a]) 
		       << " " << state1.to_string2() 
		       << " (N,Na,Nb)=" << state1.nelec() << ","
		       << state1.nelec_a() << "," << state1.nelec_b()
	               << " mag=" << abs(HijDiff.first) << endl;
	 	  inew++;
	       }
	    }
	 }
      } // ia 
*/
      // doubles
      for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
	 auto pr = tools::inverse_pair0(ijdx);
	 int i = olst[pr.first], j = olst[pr.second];
	 int ij = tools::canonical_pair0(i,j);
	 for(const auto& p : hbtab.eri.at(ij)){
	    if(p.first*cmax[idx] < eps1) break; // avoid searching all 
	    auto ab = tools::inverse_pair0(p.second);
	    int a = ab.first, b = ab.second;
	    if(state[a]==0 && state[b]==0){ // excitations
	       onstate state2(state);
	       state2[i] = 0;
	       state2[j] = 0;
	       state2[a] = 1;
	       state2[b] = 1;
	       auto search = varSpace.find(state2);
	       if(search == varSpace.end()){
	          varSpace.insert(state2);
	          space.push_back(state2);
		  if(debug){
		     cout << "   " << inew
	                  << " D(ij->ab) = " << symbol(i) << "," << symbol(j)
	                  << "->" << symbol(a) << "," << symbol(b) 
			  << " " << state2.to_string2()
			  << " (N,Na,Nb)=" << state2.nelec() << ","
			  << state2.nelec_a() << "," << state2.nelec_b()
		          << " mag=" << p.first << endl;
		     inew++;
		  }
	       }
	    }
	 } // ab
      } // ij
   } // idx
   cout << "dim0 = " << dim 
	<< " dim1 = " << space.size() 
	<< " new = " << space.size()-dim << endl;
   auto t1 = global::get_time();
   cout << "timing for sci::expand_varSpace : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}

void sci::ci_solver(vector<double>& es,
	       	    vector<vector<double>>& vs,	
		    onspace& space,
		    const input::schedule& schd, 
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   cout << "\nsci::ci_solver" << endl; 
   bool debug = true;
   auto t0 = global::get_time();

   // set up intial configurations
   unordered_set<onstate> varSpace;
   int k = int1e.sorb;
   for(const auto& det : schd.det_seeds){
      // convert det to onstate
      onstate state(k);
      for(int i : det){
         state[i] = 1;
      }
      space.push_back(state);
      varSpace.insert(state);
   }

   // set up head-bath table
   heatbath_table hbtab(int2e);

   // set up initial states
   int nsub = space.size();
   int neig = min(schd.nroots, nsub);
   vector<double> etmp(neig);
   matrix vtmp(nsub,neig);
   fock::ci_solver(etmp, vtmp, space, int2e, int1e, ecore);

   // set up initial sparseH
   product_space pspace(space);
   coupling_table ctabA(pspace.umapA);
   coupling_table ctabB(pspace.umapB);
   sparse_hamiltonian sparseH(space, pspace, ctabA, ctabB,
		   	      int2e, int1e, ecore);

   // start increment
   for(int iter=0; iter<schd.maxiter; iter++){
      cout << "\n-------------" << endl;
      cout << "iter=" << iter << " eps1=" << schd.eps1[iter] << endl;
      cout << "-------------" << endl;
      double eps1 = schd.eps1[iter];

      // print initial space here?

      // compute |cmax| for screening
      vector<double> cmax(nsub,0.0);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub; i++){
	    cmax[i] += pow(vtmp(i,j),2);
         }
      }
      transform(cmax.begin(), cmax.end(), cmax.begin(),
		[](const double& x){ return pow(x,0.5); });

      // expand 
      expand_varSpace(space, varSpace, int2e, int1e, hbtab, cmax, eps1);

      product_space pspace(space);
      coupling_table ctabA(pspace.umapA);
      coupling_table ctabB(pspace.umapB);
      sparse_hamiltonian sparseH(space, pspace, ctabA, ctabB,
   		   	         int2e, int1e, ecore);
   
      // update
      nsub = space.size();
      neig = min(schd.nroots, nsub);
      // set up Davidson solver 
      dvdsonSolver solver;
      solver.iprt = 2;
      solver.ndim = nsub;
      solver.neig = neig;
      solver.Diag = sparseH.diag.data();
      using std::placeholders::_1;
      using std::placeholders::_2;
      solver.HVec = bind(fci::get_Hx, _1, _2, cref(sparseH));
      // get initial guess
      matrix v0(solver.ndim, solver.neig);
      get_initial(space, int2e, int1e, ecore, sparseH.diag, v0);
      // solve
      vector<double> etmp1(neig);
      matrix vtmp1(nsub,neig);
      solver.solve_iter(etmp1.data(), vtmp1.data(), v0.data());
      // check convergence of SCI
      vtmp = vtmp1; // copy assignment
      
      // increment sparseH 
      
      // analysi of coefficients here!

   
   } // iter

   exit(1);

   auto t1 = global::get_time();
   cout << "timing for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
