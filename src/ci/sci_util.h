#ifndef SCI_UTIL_H
#define SCI_UTIL_H

#include <unordered_set>
#include <functional>
#include <vector>
#include <map>
#include "../core/analysis.h"
#include "../core/hamiltonian.h"
#include "../core/integral.h"
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"  // get_ortho_basis
#include "../io/input.h" // schedule
#include "fci_util.h"

namespace sci{

// type information
template <typename Tm>
struct heatbath_single{
   using type = std::vector<std::vector<float>>; 
};
template <>
struct heatbath_single<std::complex<double>>{ 
   using type = std::vector<std::vector<std::complex<float>>>; 
};

template <typename Tm>
struct heatbath_table{
public: 
   heatbath_table(const integral::two_body<Tm>& int2e,
		  const integral::one_body<Tm>& int1e){
      const bool debug = false;
      auto t0 = tools::get_time();
      std::cout << "\nheatbath_table::heatbath_table" << std::endl;
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
                  double mag = std::abs(int2e.get(i,j,p,q)); // |<ij||pq>|
                  if(mag > thresh) eri4[ij].insert(std::make_pair(mag,p*(p-1)/2+q));
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
               // crucial for using <ip||jp> rather than |<ip||jp>| 
               eri3[ij][p] = int2e.get(i,p,j,p);
            } // p
            eri3[ij][k] = int1e.get(i,j);
         } // j
      } // i
      if(debug){
	 std::cout << std::defaultfloat << std::setprecision(12);
         for(int ij=0; ij<k*(k-1)/2; ij++){
            auto pr = tools::inverse_pair0(ij);
            int i = pr.first, j = pr.second;
            std::cout << "ij=" << ij << " i,j=" << i << "," << j 
                      << " eri4[ij] size : " << eri4[ij].size() << std::endl;
            for(const auto& p : eri4[ij]){
               if(p.first > 1.e-2){
                  auto pq = tools::inverse_pair0(p.second);
        	  std::cout << "   val=" << p.first 
           	            << " -> p,q=" << pq.first << "," << pq.second 
           	            << std::endl;
               }
            }
         }
      } // debug
      auto t1 = tools::get_time();
      std::cout << "timing for heatbath_table::heatbath_table : " << std::setprecision(2) 
                << tools::get_duration(t1-t0) << " s" << std::endl;
   }
public:
   int sorb;
   // cut-off value 
   double thresh = 1.e-14; 
   // sorted by magnitude Iij[kl]=|<ij||kl>| (i>j,k>l)
   std::vector<std::multimap<float,int,std::greater<float>>> eri4; 
   // Iik[j]={<ij||kj>(i>=k),hik} for fast estimation of singles
   typename heatbath_single<Tm>::type eri3;
};

// expand variational subspace
template <typename Tm>
void expand_varSpace(fock::onspace& space, 
		     std::unordered_set<fock::onstate>& varSpace, 
		     const heatbath_table<Tm>& hbtab, 
		     const std::vector<double>& cmax, 
		     const double eps1,
		     const bool flip){
   const bool debug = false;
   auto t0 = tools::get_time();
   std::cout << "\nsci::expand_varSpace dim = " 
	     << space.size() << " eps1 = " << eps1 << std::endl;
   // assuming particle number conserving space
   fock::onstate state = space[0];
   int no = state.nelec(), k = state.size(), nv = k - no;
   std::vector<int> olst(no), vlst(nv);
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
	 std::cout << " i=" << idx << " " << state
	           << " (N,Na,Nb)=" << state.nelec()
	           << "," << state.nelec_a() << "," << state.nelec_b()
	           << " cmax=" << cmax[idx] << std::endl;	 
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
	 if(std::abs(HijS_bound)*cmax[idx] > eps1){
	    fock::onstate state1(state);
	    state1[i] = 0;
	    state1[a] = 1;
	    auto search = varSpace.find(state1);
	    if(search == varSpace.end()){
	       if(debug){
		  std::cout << "   " << ns
	                    << " S(i->a) = " << fock::symbol(i)
		            << "->" << fock::symbol(a) 
		            << " " << state1
		            << " (N,Na,Nb)=" << state1.nelec() << ","
		            << state1.nelec_a() << "," << state1.nelec_b()
	                    << " mag=" << std::abs(HijS_bound) << " " << cmax[idx] 
			    << std::endl;
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
   std::cout << "no. of singles = " << ns << " timing : " << std::setprecision(2) 
	     << tools::get_duration(ts-t0) << " s" << std::endl;
   
   // doubles
   int nd = 0;
   for(int idx=0; idx<dim; idx++){
      // select |Dj> if |<Dj|H|Di>cmax[i]|>eps1 && |Dj> is not in V 
      state = space[idx];
      state.get_olst(olst.data());
      state.get_vlst(vlst.data());
      if(debug){
	 std::cout << " i=" << idx << " " << state
	           << " (N,Na,Nb)=" << state.nelec()
	           << "," << state.nelec_a() << "," << state.nelec_b()
	           << " cmax=" << cmax[idx] << std::endl;
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
	       fock::onstate state2(state);
	       state2[i] = 0;
	       state2[j] = 0;
	       state2[a] = 1;
	       state2[b] = 1;
	       auto search = varSpace.find(state2);
	       if(search == varSpace.end()){
		  if(debug){
		     std::cout << "   " << nd
	                       << " D(ij->ab) = " << fock::symbol(i) << "," << fock::symbol(j)
	                       << "->" << fock::symbol(a) << "," << fock::symbol(b) 
			       << " " << state2
			       << " (N,Na,Nb)=" << state2.nelec() << ","
			       << state2.nelec_a() << "," << state2.nelec_b()
		               << " mag=" << p.first 
			       << std::endl;
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
   std::cout << "no. of doubles = " << nd << " timing : " << std::setprecision(2) 
	     << tools::get_duration(td-ts) << " s" << std::endl;

   std::cout << "dim = " << dim << " new = " << space.size()-dim 
	     << " total = " << space.size() << std::endl;
   auto t1 = tools::get_time();
   std::cout << "timing for sci::expand_varSpace : " << std::setprecision(2) 
	     << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // sci

#endif
