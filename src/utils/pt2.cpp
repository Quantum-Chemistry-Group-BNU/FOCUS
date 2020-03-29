#include "pt2.h"
#include "../core/hamiltonian.h"
#include "../core/tools.h"
#include <unordered_map>
#include <tuple>

using namespace std;
using namespace sci;
using namespace fock;

// selected CI procedure
void sci::pt2_solver(const input::schedule& schd, 
		     const double e0,
	       	     const vector<double>& v0,
		     const onspace& space,
	       	     const integral::two_body& int2e,
	       	     const integral::one_body& int1e,
	       	     const double ecore){
   cout << "\nsci::pt2_solver" << endl; 
   bool debug = true;
   auto t0 = global::get_time();
  
   // set up head-bath table
   heatbath_table hbtab(int2e, int1e);
   
   // set up hash table
   unordered_set<onstate> varSpace;
   for(const auto& det : space){ 
      varSpace.insert(det);
   }

   // assuming particle number conserving space
   onstate state = space[0];
   int no = state.nelec(), k = state.size(), nv = k - no;
   vector<int> olst(no), vlst(nv);
   int nsingles = no*nv;
 
   unordered_map<onstate,double> pt2Space; 
   vector<pair<onstate,double>> pt2; 
   const double eps2 = -1.e-10;
   // select |Dj> if |<Dj|H|Di>cmax[i]|>eps2 && |Dj> is not in V 
   int dim = space.size();
   for(int idx=0; idx<dim; idx++){
      state = space[idx];
      state.get_olst(olst.data());
      state.get_vlst(vlst.data());
      for(int ia=0; ia<nsingles; ia++){
         int ix = ia%no, ax = ia/no;
	 int i = olst[ix], a = vlst[ax];
	 onstate state1(state);
	 state1[i] = 0;
	 state1[a] = 1;
	 auto search = varSpace.find(state1);
	 if(search == varSpace.end()){
	    auto pr = fock::get_HijS(state1,state,int2e,int1e);
	    double Hc = pr.first*v0[idx];
	    if(abs(Hc) > eps2) pt2.emplace_back(state1,Hc);
	      
	    auto search1 = pt2Space.find(state1);
	    if(search1 == pt2Space.end()){
	       pt2Space[state1] = Hc;
	    }else{
	       search1->second += Hc;
	    }

	 } 
      } // ia 
      for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
	 auto pr = tools::inverse_pair0(ijdx);
	 int i = olst[pr.first], j = olst[pr.second];
	 int ij = tools::canonical_pair0(i,j);
	 for(const auto& p : hbtab.eri4.at(ij)){
	    if(p.first*abs(v0[idx]) < eps2) break; // avoid searching all doubles
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
  	          auto pr = fock::get_HijD(state2,state,int2e,int1e);
  	    	  double Hc = pr.first*v0[idx];
	          if(abs(Hc) > eps2) pt2.emplace_back(state2,Hc);

           	  auto search2 = pt2Space.find(state2);
           	  if(search2 == pt2Space.end()){
           	     pt2Space[state2] = Hc;
           	  }else{
           	     search2->second += Hc;
           	  }

	       }
	    }
	 } // ab
      } // ij
   } // idx

   stable_sort(pt2.begin(), pt2.end(),
	       [](pair<onstate,double> x1, pair<onstate,double> x2){
	           return x1.first < x2.first; });
   
   // collect all contributions
   int pdim = pt2.size();
   cout << "\npt2 summary:" << endl;  
   cout << "vdim = " << setw(20) << dim << endl;
   cout << "pdim = " << setw(20) << pdim << endl;
   // init
   double ept2 = 0.0;
   onstate state_a = pt2[0].first;
   double va0 = pt2[0].second;
   double ea = fock::get_Hii(state_a,int2e,int1e)+ecore;
   for(int i=1; i<pdim; i++){
      if(pt2[i].first == state_a){
	 va0 += pt2[i].second; // += <a|H|i>ci
      }else{
	 // finish accumulation for a:
	 ept2 += pow(va0,2)/(e0-ea);
	 // update
	 state_a = pt2[i].first;
	 va0 = pt2[i].second;
	 ea = fock::get_Hii(state_a,int2e,int1e)+ecore;
      }
   }
   ept2 += pow(va0,2)/(e0-ea);
   
   // print summary
   cout << "eCI  = " << fixed << setw(20) << setprecision(12) << e0 << endl; 
   cout << "ePT2 = " << fixed << setw(20) << setprecision(12) << ept2 << endl;
   cout << "etot = " << fixed << setw(20) << setprecision(12) << e0+ept2 << endl;

   double e2 = 0.0;
   for(const auto& pr : pt2Space){
      ea = fock::get_Hii(pr.first,int2e,int1e)+ecore;
      va0 = pr.second;
      e2 += pow(va0,2)/(e0-ea); 
   }
   cout << "dim= " << pt2Space.size() << endl;
   cout << "e2 = " << e2 << endl;

   auto t1 = global::get_time();
   cout << "timing for sci::pt2_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
