#ifndef SCI_PT2_H
#define SCI_PT2_H

#include <unordered_map>
#include "sci_util.h"

namespace sci{

// PT2 for single state 	
template <typename Tm>
void pt2_solver(const input::schedule& schd,
	        const double e0,
	        const std::vector<Tm>& v0,
	        const fock::onspace& space,
	        const integral::two_body<Tm>& int2e,
	        const integral::one_body<Tm>& int1e,
	        const double ecore){
   const bool debug = true;
   auto t0 = tools::get_time();
   std::cout << "\nsci::pt2_solver" << std::endl;  
   // set up head-bath table
   heatbath_table<Tm> hbtab(int2e, int1e);
   // set up hash table
   std::unordered_set<fock::onstate> varSpace;
   for(const auto& det : space){ 
      varSpace.insert(det);
   }
   // assuming particle number conserving space
   fock::onstate state = space[0];
   int no = state.nelec(), k = state.size(), nv = k - no;
   std::vector<int> olst(no), vlst(nv);
   std::unordered_map<fock::onstate,Tm> pt2Space; 
   // select |Dj> if |<Dj|H|Di>cmax[i]|>eps2 && |Dj> is not in V 
   int nsingles = no*nv;
   int vdim = space.size();
   for(int idx=0; idx<vdim; idx++){
      state = space[idx];
      state.get_olst(olst.data());
      state.get_vlst(vlst.data());
      // singles
      for(int ia=0; ia<nsingles; ia++){
         int ix = ia%no, ax = ia/no;
	 int i = olst[ix], a = vlst[ax];
	 fock::onstate state1(state);
	 state1[i] = 0;
	 state1[a] = 1;
	 auto search = varSpace.find(state1);
	 if(search == varSpace.end()){
	    auto pr = fock::get_HijS(state1,state,int2e,int1e);
	    Tm Hc = pr.first*v0[idx];
	    if(std::abs(Hc)<schd.eps2) continue;
	    auto search = pt2Space.find(state1);
	    if(search == pt2Space.end()){
	       pt2Space[state1] = Hc;
	    }else{
	       search->second += Hc; // accumulate
	    }
	 } 
      } // ia 
      // doubles
      for(int ijdx=0; ijdx<no*(no-1)/2; ijdx++){
	 auto pr = tools::inverse_pair0(ijdx);
	 int i = olst[pr.first], j = olst[pr.second];
	 int ij = tools::canonical_pair0(i,j);
	 for(const auto& p : hbtab.eri4.at(ij)){
	    if(p.first*std::abs(v0[idx]) < schd.eps2) break; // avoid searching all doubles
	    auto ab = tools::inverse_pair0(p.second);
	    int a = ab.first, b = ab.second;
	    if(state[a]==0 && state[b]==0){ // if true double excitations
	       fock::onstate state2(state);
	       state2[i] = 0;
	       state2[j] = 0;
	       state2[a] = 1;
	       state2[b] = 1;
	       auto search = varSpace.find(state2);
	       if(search == varSpace.end()){
  	          auto pr = fock::get_HijD(state2,state,int2e,int1e);
  	    	  Tm Hc = pr.first*v0[idx];
	          if(std::abs(Hc)<schd.eps2) continue;
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
  
   // collect all contributions
   const int nmax = 50;
   int pdim = pt2Space.size();
   double e2 = 0.0, z2 = 1.0, S_PT = 0.0;
   std::vector<double> e2max(nmax,0.0);   // e2a
   std::vector<fock::onstate> vmax(nmax); // onstate |Da>
   for(const auto& pr : pt2Space){
      Tm va0 = pr.second;
      double ea = fock::get_Hii(pr.first,int2e,int1e)+ecore;
      double e2tmp = std::norm(va0)/(e0-ea);
      double pa = std::norm(va0/(e0-ea));
      e2 += e2tmp;
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

   std::cout << "\nlargest " << nmax << " e2a contributions:" << std::endl;
   std::cout << std::fixed << std::setprecision(12);
   double e2sum = 0.0;
   for(int i=0; i<nmax; i++){
      double ea = fock::get_Hii(vmax[i],int2e,int1e)+ecore;
      Tm va0 = pt2Space[vmax[i]];
      std::cout << " i= " << std::noshowpos << std::setw(3) << i 
	   << " e2a= " << std::setprecision(12) << std::showpos << e2max[i] 
	   << " D0a= " << std::setprecision(4) << std::showpos << e0-ea
	   << " Ha0= " << std::setprecision(4) << std::showpos << va0
	   << " Ha0/D0a= " << std::setprecision(4) << std::showpos << va0/(e0-ea)
	   << std::endl;
      e2sum += e2max[i];
   } // i
   std::cout << "e2sum= " << std::setprecision(12) << e2sum << std::endl;
   std::cout << "e2tot= " << std::setprecision(12) << e2 << std::endl;
   std::cout << "per= " << std::setprecision(1) << std::noshowpos << e2sum/e2*100 << std::endl;

   std::cout << "\npt2 summary: eps2=" << std::defaultfloat << schd.eps2 << std::endl;  
   std::cout << "vdim = " << std::setw(20) << vdim << std::endl;
   std::cout << "pdim = " << std::setw(20) << pdim << std::endl;
   // diagonal entropy
   double S_CI = fock::coeff_entropy(v0);
   double S_tot = (S_CI+S_PT)/z2 + log2(z2);
   std::cout << "z2   = " << std::fixed << std::setw(20) << std::setprecision(12) << z2 << std::endl;
   std::cout << "eCI  = " << std::fixed << std::setw(20) << std::setprecision(12) << e0 
       	     << "     S_CI = " << S_CI << std::endl; 
   std::cout << "ePT2 = " << std::fixed << std::setw(20) << std::setprecision(12) << e2 
	     << "     S_PT = " << S_PT << std::endl;
   std::cout << "etot = " << std::fixed << std::setw(20) << std::setprecision(12) << e0+e2 
	     << "     Stot = " << S_tot << std::endl;

   auto t1 = tools::get_time();
   std::cout << "timing for sci::pt2_solver : " << std::setprecision(2) 
	     << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // sci

#endif
