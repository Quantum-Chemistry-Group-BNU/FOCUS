#include <unordered_map>
#include <tuple>
#include "../core/hamiltonian.h"
#include "../core/analysis.h"
#include "../core/tools.h"
#include "sci.h"

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
   bool debug = true;
   auto t0 = global::get_time();
   cout << "\nsci::pt2_solver" << endl; 
  
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
   // select |Dj> if |<Dj|H|Di>cmax[i]|>eps2 && |Dj> is not in V 
   int vdim = space.size();
   for(int idx=0; idx<vdim; idx++){
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
	    if(abs(Hc)<schd.eps2) continue;
	    auto search = pt2Space.find(state1);
	    if(search == pt2Space.end()){
	       pt2Space[state1] = Hc;
	    }else{
	       search->second += Hc; // accumulate
	    }
	 } 
      } // ia 
      for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
	 auto pr = tools::inverse_pair0(ijdx);
	 int i = olst[pr.first], j = olst[pr.second];
	 int ij = tools::canonical_pair0(i,j);
	 for(const auto& p : hbtab.eri4.at(ij)){
	    if(p.first*abs(v0[idx]) < schd.eps2) break; // avoid searching all doubles
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
	          if(abs(Hc)<schd.eps2) continue;
           	  auto search = pt2Space.find(state2);
           	  if(search == pt2Space.end()){
           	     pt2Space[state2] = Hc;
           	  }else{
           	     search->second += Hc;
           	  }
	       }
	    }
	 } // ab
      } // ij
   } // idx
  
   // redefined energy following p-DMRG
   double e0wt = 0.0;
   for(int i=0; i<vdim; i++){
      double ei = fock::get_Hii(space[i],int2e,int1e)+ecore;
      e0wt += ei*pow(v0[i],2);
   }	
   double e2av = 0.0;

   // collect all contributions
   int pdim = pt2Space.size();
   double e2 = 0.0, z2 = 1.0, S_PT = 0.0;
   int nmax = 100;
   vector<double> e2max(nmax,0.0);
   vector<onstate> vmax(nmax);
   for(const auto& pr : pt2Space){
      double ea = fock::get_Hii(pr.first,int2e,int1e)+ecore;
      double va0 = pr.second;
      double e2tmp = pow(va0,2)/(e0-ea);
      e2 += e2tmp;
      e2av += pow(va0,2)/(0.5*(e0+e0wt)-ea);
      double pa = pow(va0/(e0-ea),2);
      z2 += pa;
      if(pa > 1.e-12) S_PT += -pa*log2(pa);
      // check whether smaller e2tmp has been found
      for(int i=0; i<nmax; i++){
	 if(e2tmp < e2max[i]){
	    for(int j=nmax-1; j>i; j--){
	       e2max[j] = e2max[j-1];
	       vmax[j] = vmax[j-1];
	    }
	    e2max[i] = e2tmp;
	    vmax[i] = pr.first;
	    break;
	 }
      } // i
   }

   cout << "\nstatistics for individual contributions:" << endl;
   cout << fixed << setprecision(12);
   double e2sum = 0.0;
   for(int i=0; i<nmax; i++){
      double ea = fock::get_Hii(vmax[i],int2e,int1e)+ecore;
      double va0 = pt2Space[vmax[i]];
      cout << " i= " << noshowpos << setw(3) << i 
	   << " e2= " << setprecision(12) << showpos << e2max[i] 
	   << " D0a= " << setprecision(4) << showpos << e0-ea
	   << " Ha0= " << setprecision(4) << showpos << va0
	   << " Ha0/D0a= " << setprecision(4) << showpos << va0/(e0-ea)
	   << endl;
      e2sum += e2max[i];
   } // i
   cout << "e2sum= " << e2sum << endl;
   cout << "e2tot= " << e2 << endl;
   cout << "per= " << setprecision(1) << noshowpos << e2sum/e2*100 << endl;

   cout << "\npt2 summary: eps2=" << defaultfloat << schd.eps2 << endl;  
   cout << "vdim = " << setw(20) << vdim << endl;
   cout << "pdim = " << setw(20) << pdim << endl;
   // diagonal entropy
   double S_CI = fock::coeff_entropy(v0);
   double S_tot = (S_CI+S_PT)/z2 + log2(z2);
   cout << "z2   = " << fixed << setw(20) << setprecision(12) << z2 << endl;
   cout << "eCI  = " << fixed << setw(20) << setprecision(12) << e0 
	<< "     S_CI = " << S_CI << endl; 
   cout << "ePT2 = " << fixed << setw(20) << setprecision(12) << e2 
	<< "     S_PT = " << S_PT << endl;
   cout << "etot = " << fixed << setw(20) << setprecision(12) << e0+e2 
	<< "     Stot = " << S_tot << endl;
   // modified e0
   cout << endl;
   cout << "e0wt = " << fixed << setw(20) 
	   	     << setprecision(12) << e0wt << endl;
   cout << "ePT2h= " << fixed << setw(20) 
		     << setprecision(12) << e2av << endl;
   cout << "etot = " << fixed << setw(20) 
	    	     << setprecision(12) << e0+e2av << endl;

   auto t1 = global::get_time();
   cout << "timing for sci::pt2_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
