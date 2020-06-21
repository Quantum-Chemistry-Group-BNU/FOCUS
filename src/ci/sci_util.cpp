#include "sci_util.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;
using namespace sci;

// expand variational subspace
void sci::expand_varSpace(onspace& space, 
			  unordered_set<onstate>& varSpace,
		          const heatbath_table& hbtab, 
			  const vector<double>& cmax, 
			  const double eps1,
			  const bool flip){
   const bool debug = false;
   auto t0 = tools::get_time();
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
	 auto HijS_bound = hbtab.eri3[pq][k]; // hai
	 for(int jx=0; jx<no; jx++){
            int j = olst[jx];
	    HijS_bound += hbtab.eri3[pq][j];
	 } // <aj||ij>
	 // heat-bath check
	 if(abs(HijS_bound)*cmax[idx] > eps1){
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
	               << " mag=" << abs(HijS_bound) << " " << cmax[idx]<< endl;
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
   auto ts = tools::get_time();
   cout << "no. of singles = " << ns << " timing : " << setprecision(2) 
	<< tools::get_duration(ts-t0) << " s" << endl;
   
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
   auto td = tools::get_time();
   cout << "no. of doubles = " << nd << " timing : " << setprecision(2) 
	<< tools::get_duration(td-ts) << " s" << endl;

   cout << "dim = " << dim << " new = " << space.size()-dim 
	<< " total = " << space.size() << endl;
   auto t1 = tools::get_time();
   cout << "timing for sci::expand_varSpace : " << setprecision(2) 
	<< tools::get_duration(t1-t0) << " s" << endl;
}
