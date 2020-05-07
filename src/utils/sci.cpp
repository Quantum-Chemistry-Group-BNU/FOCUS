#include "../settings/global.h"
#include "../core/hamiltonian.h"
#include "../core/linalg.h"
#include "../core/onstate.h"
#include "../core/dvdson.h"
#include "../core/tools.h"
#include "../core/analysis.h"
#include "sci.h"
#include "fci.h"
#include "fci_rdm.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;
using namespace sci;

heatbath_table::heatbath_table(const integral::two_body& int2e,
			       const integral::one_body& int1e){
   bool debug = false;
   auto t0 = global::get_time();
   cout << "\nheatbath_table::heatbath_table" << endl;

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
	 eri3[ij].resize(k+1);
	 for(int p=0; p<k; p++){
	    // <ip||jp> = [ij|pp] - [ip|pj] (i>=j)
	    eri3[ij][p] = int2e.get(i,j,p,p) - int2e.get(i,p,p,j); 
	 } // p
	 eri3[ij][k] = int1e.get(i,j);
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
		          const heatbath_table& hbtab, 
			  const vector<double>& cmax, 
			  const double eps1,
			  const bool flip){
   bool debug = false;
   auto t0 = global::get_time();
   cout << "\nsci::expand_varSpace dim = " 
	<< space.size() << " eps1 = " << eps1 << endl;

   // assuming particle number conserving space
   onstate state = space[0];
   int no = state.nelec(), k = state.size(), nv = k - no;
   vector<int> olst(no), vlst(nv);
   int nsingles = no*nv;
   int dim = space.size();
   
   // singles
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
      for(int ia=0; ia<nsingles; ia++){
         int ix = ia%no, ax = ia/no;
	 int i = olst[ix], a = vlst[ax];
	 // direct computation of HijS using eri3 [fast]
	 int p = std::max(i,a), q = std::min(i,a), pq = p*(p+1)/2+q;
	 double Hij = hbtab.eri3[pq][k]; // hai
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
	       // flip
	       if(flip){
	          auto state1f = state1.flip();
	          auto search1 = varSpace.find(state1f);
	          if(search1 == varSpace.end()){
	             varSpace.insert(state1f);
	             space.push_back(state1f);
 	             ns++;
	          }
	       }
	    }
	 } 
      } // ia 
   } // idx
   auto ts = global::get_time();
   cout << "no. of singles = " << ns << " timing : " << setprecision(2) 
	<< global::get_duration(ts-t0) << " s" << endl;
   
   // doubles
   int nd = 0;
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
	          // flip
		  if(flip){
	             auto state2f = state2.flip();
	             auto search2 = varSpace.find(state2f);
	             if(search2 == varSpace.end()){
	                varSpace.insert(state2f);
	                space.push_back(state2f);
 	                nd++;
	             }
		  }
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

// prepare intial solution
void sci::get_initial(vector<double>& es,
		      matrix& vs,
		      onspace& space, 
		      unordered_set<onstate>& varSpace, 
		      const heatbath_table& hbtab, 
		      const input::schedule& schd, 
		      const integral::two_body& int2e, 
		      const integral::one_body& int1e, 
		      const double ecore){
   cout << "\nsci::get_initial" << endl;
   // space = {|Di>}
   int k = int1e.sorb;
   for(const auto& det : schd.det_seeds){
      // convert det to onstate
      onstate state(k);
      for(int i : det){
         state[i] = 1;
      }
      // search first
      auto search = varSpace.find(state);
      if(search == varSpace.end()){
	 varSpace.insert(state);
	 space.push_back(state);
      }
      // flip determinant for S=0
      if(schd.flip){
         auto state1 = state.flip();
         auto search1 = varSpace.find(state1);
         if(search1 == varSpace.end()){
            space.push_back(state1);
            varSpace.insert(state1);
         }
      }
   }
   // print
   cout << "energies for reference states:" << endl;
   cout << defaultfloat << setprecision(12);
   int nsub = space.size();
   for(int i=0; i<nsub; i++){
      cout << "i = " << i << " state = " << space[i].to_string2() 
	   << " e = " << fock::get_Hii(space[i],int2e,int1e)+ecore 
	   << endl;
   }
   // selected CISD space
   double eps1 = schd.eps0;
   vector<double> cmax(nsub,1.0);
   expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.flip);
   nsub = space.size();
   // set up initial states
   if(schd.nroots > nsub){
      cout << "error in sci::ci_solver: subspace is too small!" << endl;
      exit(1);
   }
   auto vsol = fock::get_Ham(space, int2e, int1e, ecore);
   vector<double> esol(nsub);
   eigen_solver(vsol, esol);
   // save
   int neig = schd.nroots;
   es.resize(neig);
   matrix vtmp(nsub, neig);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nsub; i++){
	 vtmp(i,j) = vsol(i,j);
      }
      es[j] = esol[j];
   }
   vs = vtmp;
   // print
   cout << setprecision(12);
   for(int i=0; i<neig; i++){
      cout << "i = " << i << " e = " << es[i] << endl; 
   }
}

// selected CI procedure
void sci::ci_solver(const input::schedule& schd, 
	            sparse_hamiltonian& sparseH,
		    vector<double>& es,
	       	    vector<vector<double>>& vs,	
		    onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   bool debug = false;
   auto t0 = global::get_time();
   cout << "\nsci::ci_solver" << endl; 
   
   // set up head-bath table
   heatbath_table hbtab(int2e, int1e);

   // set up intial configurations
   vector<double> esol;
   matrix vsol;
   unordered_set<onstate> varSpace;
   get_initial(esol, vsol, space, varSpace, 
	       hbtab, schd, int2e, int1e, ecore);
   
   // set up auxilliary data structure   
   product_space pspace;
   coupling_table ctabA, ctabB;
   // build initial
   pspace.get_pspace(space);
   ctabA.get_C11(pspace.spaceA);
   ctabB.get_C11(pspace.spaceB);
   sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
   		   	   int2e, int1e, ecore);

   // start increment
   bool ifconv = false;
   int nsub = space.size(); 
   int neig = schd.nroots;
   for(int iter=0; iter<schd.maxiter; iter++){
      cout << "\n---------------------" << endl;
      cout << "iter=" << iter << " eps1=" << schd.eps1[iter] << endl;
      cout << "---------------------" << endl;
      double eps1 = schd.eps1[iter];

      // compute |cmax| for screening
      vector<double> cmax(nsub,0.0);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub; i++){
	    cmax[i] += pow(vsol(i,j),2);
         }
      }
      transform(cmax.begin(), cmax.end(), cmax.begin(),
		[neig](const double& x){ return pow(x/neig,0.5); });

      // expand 
      expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.flip);
      int nsub0 = nsub;
      nsub = space.size();

      // update auxilliary data structure 
      pspace.get_pspace(space, nsub0);
      ctabA.get_C11(pspace.spaceA, pspace.dimA0);
      ctabB.get_C11(pspace.spaceB, pspace.dimB0);
      sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
        	   	      int2e, int1e, ecore, nsub0);

      // set up Davidson solver 
      dvdsonSolver solver;
      solver.iprt = 2;
      solver.crit_v = schd.crit_v;
      solver.maxcycle = schd.maxcycle;
      solver.ndim = nsub;
      solver.neig = neig;
      solver.Diag = sparseH.diag.data();
      using std::placeholders::_1;
      using std::placeholders::_2;
      solver.HVec = bind(fci::get_Hx, _1, _2, cref(sparseH));

      // copy previous initial guess
      matrix v0(nsub, neig);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub0; i++){
            v0(i,j) = vsol(i,j);
	 }
      }
      
      // solve
      vector<double> esol1(neig);
      matrix vsol1(nsub,neig);
      solver.solve_iter(esol1.data(), vsol1.data(), v0.data());

      // check convergence of SCI
      vector<bool> conv(neig);
      cout << endl;
      for(int i=0; i<neig; i++){
	 conv[i] = abs(esol1[i]-esol[i])<schd.deltaE; 
	 vector<double> vtmp(vsol1.col(i),vsol1.col(i)+nsub);
         double SvN = coeff_entropy(vtmp); 
	 cout << "iter=" << iter
	      << " eps1=" << scientific << setprecision(2) << schd.eps1[iter]
	      << " nsub=" << nsub 
	      << " i=" << i 
	      << " e=" << defaultfloat << setprecision(12) << esol1[i] 
	      << " de=" << scientific << setprecision(2) << esol1[i]-esol[i] 
	      << " conv=" << conv[i] 
	      << " SvN=" << SvN
	      << endl;
      }
      esol = esol1;
      vsol = vsol1;
      ifconv = (count(conv.begin(), conv.end(), true) == neig);
      if(iter>=schd.miniter && ifconv){
	 cout << "convergence is achieved!" << endl;
	 break;
      }
   } // iter
   if(!ifconv){
      cout << "convergence failure: out of maxiter=" << schd.maxiter << endl;
   }

   // copy results
   copy_n(esol.begin(), neig, es.begin());
   for(int i=0; i<neig; i++){
      vs[i].resize(nsub);
      copy_n(vsol.col(i), nsub, vs[i].begin());
   }

   auto t1 = global::get_time();
   cout << "timing for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}

void sci::ci_truncate(onspace& space,
	  	      vector<vector<double>>& vs,	
		      const int maxdets){
   cout << "\nsci::ci_truncate maxdets=" << maxdets << endl;
   int nsub = space.size();
   int neig = vs.size();
   int nred = min(nsub,maxdets);
   cout << "reduction from " << nsub << " to " << nred << " dets" << endl;
   // select important basis
   vector<double> cmax(nsub,0.0);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nsub; i++){
	 cmax[i] += pow(vs[j][i],2);
      }
   }
   auto index = tools::sort_index(cmax); 
   // orthogonalization
   vector<double> vtmp(nred*neig);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nred; i++){
	 vtmp[i+nred*j] = vs[j][index[i]];
      }
   }
   int nindp = linalg::get_ortho_basis(nred,neig,vtmp);
   if(nindp != neig){
      cout << "error: thresh is too large for ci_truncate!" << endl;
      cout << "nindp,neig=" << nindp << "," << neig << endl;
      exit(1);
   }
   // copy basis and coefficients
   onspace space2(nred);
   for(int i=0; i<nred; i++){
      space2[i] = space[index[i]];	
   }
   vector<vector<double>> vs2(neig);
   for(int j=0; j<neig; j++){
      vs2[j].resize(nred);
      copy(&vtmp[nred*j],&vtmp[nred*j]+nred,vs2[j].begin());
   }
   // check
   for(int j=0; j<neig; j++){
      vector<double> vec(nred);
      for(int i=0; i<nred; i++){
	 vec[i] = vs[j][index[i]];
      }
      double ova = ddot(nred,vs2[j].data(),vec.data());
      cout << "iroot=" << j << " ova=" 
	   << setprecision(12) << ova << endl; 
   }
   space = space2;
   vs = vs2;
}
