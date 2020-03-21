#include "../settings/global.h"
#include "../core/onstate.h"
#include "../core/dvdson.h"
#include "../core/tools.h"
#include "fci.h"
#include "sci.h"

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
   eri4.resize(k*(k-1)/2);
   for(int i=0; i<k; i++){
      for(int j=0; j<i; j++){
	 int ij = i*(i-1)/2+j;
	 for(int p=0; p<k; p++){
	    if(p == i || p == j) continue; // guarantee true double excitations
 	    for(int q=0; q<p; q++){
	       if(q == i || q == j) continue;
	       // <ij||pq> = [ip|jq] - [iq|jp] (i>j, p>q)
	       double mag = abs(int2e.get(i,p,j,q) - int2e.get(i,q,j,p));
	       if(mag > thresh){
	          eri4[ij].insert(make_pair(mag,p*(p-1)/2+q));
	       }
	    } // q
	 } // p
      } // j
   } // i
   eri3.resize(k*(k+1)/2);
   for(int i=0; i<k; i++){
      for(int j=0; j<=i; j++){
	 int ij = i*(i+1)/2+j;
	 eri3[ij].resize(k);
	 for(int p=0; p<k; p++){
	    // <ip||jp> = [ij|pp] - [ip|pj] (i>=j)
	    eri3[ij][p] = int2e.get(i,j,p,p) - int2e.get(i,p,p,j); 
	 } // p
      } // j
   } // i
   if(debug){
      cout << defaultfloat << setprecision(12);
      for(int ij=0; ij<k*(k-1)/2; ij++){
         auto pr = tools::inverse_pair0(ij);
	 int i = pr.first, j = pr.second;
         cout << "ij=" << ij << " i,j=" << i << "," << j 
	      << " eri4[ij] size : " << eri4[ij].size() << endl;
	 for(const auto& p : eri4[ij]){
	    if(p.first > 1.e-2){
               auto pq = tools::inverse_pair0(p.second);
     	       cout << "   val=" << p.first 
		    << " -> p,q=" << pq.first << "," << pq.second 
		    << endl;
	    }
	 }
      }
   } // debug
   auto t1 = global::get_time();
   cout << "timing for heatbath_table::heatbath_table : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
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

   // assuming particle number conserving space
   onstate state = space[0];
   int no = state.nelec();
   int nv = state.size() - no;
   vector<int> olst(no), vlst(nv);
   int nsingles = no*nv;
   int dim = space.size();
   
   // loop over each det |Di> in V
   int ns = 0;
   for(int idx=0; idx<dim; idx++){
      // select |Dj> if |<Dj|H|Di>cmax[i]|>eps1 && |Dj> is not in V 
      state = space[idx];
      state.get_olst(olst.data());
      state.get_vlst(vlst.data());
      if(debug){
	 cout << " i=" << idx << " " << state.to_string2() 
	      << " (N,Na,Nb)=" << state.nelec()
	      << "," << state.nelec_a() << "," << state.nelec_b()
	      << " cmax=" << cmax[idx] << endl;	 
      }
      // singles
      for(int ia=0; ia<nsingles; ia++){
         int ix = ia%no, ax = ia/no;
	 int i = olst[ix], a = vlst[ax];
	 // direct computation of HijS using eri3 [fast]
	 int p = std::max(i,a), q = std::min(i,a);
         int pq = p*(p+1)/2+q;
	 double Hij = int1e.get(p,q); // hai 
	 for(int jx=0; jx<no; jx++){
            int j = olst[jx];
	    Hij += hbtab.eri3[pq][j];
	 } // <aj||ij>
	 // heat-bath check
	 if(abs(Hij)*cmax[idx] > eps1){
	    onstate state1(state);
	    state1[i] = 0;
	    state1[a] = 1;
	    auto search = varSpace.find(state1);
	    if(search == varSpace.end()){
	       if(debug){
		  cout << "   " << ns
	               << " S(i->a) = " << symbol(i)
		       << "->" << symbol(a) 
		       << " " << state1.to_string2() 
		       << " (N,Na,Nb)=" << state1.nelec() << ","
		       << state1.nelec_a() << "," << state1.nelec_b()
	               << " mag=" << abs(Hij) << " " << cmax[idx]<< endl;
	       }
	       varSpace.insert(state1);
	       space.push_back(state1);
 	       ns++;
	    }
	 } 
      } // ia 
   } // idx
   auto ts = global::get_time();
   cout << "no. of singles = " << ns << " timing : " << setprecision(2) 
	<< global::get_duration(ts-t0) << " s" << endl;

   int nd = 0;
   for(int idx=0; idx<dim; idx++){
      // select |Dj> if |<Dj|H|Di>cmax[i]|>eps1 && |Dj> is not in V 
      state = space[idx];
      state.get_olst(olst.data());
      state.get_vlst(vlst.data());
      // doubles
      for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
	 auto pr = tools::inverse_pair0(ijdx);
	 int i = olst[pr.first], j = olst[pr.second];
	 int ij = tools::canonical_pair0(i,j);
	 for(const auto& p : hbtab.eri4.at(ij)){
	    if(p.first*cmax[idx] < eps1) break; // avoid searching all doubles
	    auto ab = tools::inverse_pair0(p.second);
	    int a = ab.first, b = ab.second;
	    if(state[a]==0 && state[b]==0){ // if true double excitations
	       onstate state2(state);
	       state2[i] = 0;
	       state2[j] = 0;
	       state2[a] = 1;
	       state2[b] = 1;
	       auto search = varSpace.find(state2);
	       if(search == varSpace.end()){
		  if(debug){
		     cout << "   " << nd
	                  << " D(ij->ab) = " << symbol(i) << "," << symbol(j)
	                  << "->" << symbol(a) << "," << symbol(b) 
			  << " " << state2.to_string2()
			  << " (N,Na,Nb)=" << state2.nelec() << ","
			  << state2.nelec_a() << "," << state2.nelec_b()
		          << " mag=" << p.first << endl;
		  }
	          varSpace.insert(state2);
	          space.push_back(state2);
		  nd++;
	       }
	    }
	 } // ab
      } // ij
   } // idx
   auto td = global::get_time();
   cout << "no. of doubles = " << nd << " timing : " << setprecision(2) 
	<< global::get_duration(td-ts) << " s" << endl;

   cout << "dim = " << dim << " new = " << space.size()-dim 
	<< " total = " << space.size() << endl;
   auto t1 = global::get_time();
   cout << "timing for sci::expand_varSpace : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}

// selected CI procedure
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
/*
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
      solver.crit_v = 1.e-4;
      solver.crit_e = 1.e-10;
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
      etmp.resize(neig);
      matrix vtmp1(nsub,neig);
      solver.solve_iter(etmp.data(), vtmp1.data(), v0.data());
      // check convergence of SCI
      vtmp = vtmp1; // copy assignment
      
      // increment sparseH 
      
      // analysi of coefficients here!

   
   } // iter

   // copy results
   copy_n(etmp.begin(), neig, es.begin());
   for(int i=0; i<neig; i++){
      vs[i].resize(nsub);
      copy_n(vtmp.col(i), nsub, vs[i].begin());
   }
   cout << space.size() << endl;
   cout << "nsub=" << nsub << endl;
*/
   auto t1 = global::get_time();
   cout << "timing for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
