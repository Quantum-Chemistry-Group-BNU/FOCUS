#include <iostream>
#include <cassert>
#include <string>
#include <algorithm>
#include <functional>
#include "onspace.h"

using namespace std;
using namespace fock;
using namespace linalg;

void fock::check_space(onspace& space){
   cout << "\nfock::check_space dim=" << space.size() << endl; 
   for(size_t i=0; i<space.size(); i++){
       cout << "i=" << i << " : " << space[i] << endl;
   }
}

onspace fock::get_fci_space(const int k, const int n){
   //cout << "\nfock::get_fci_space (k,n)=" << k << "," << n << endl; 
   assert(k >= n);
   onspace space;
   string s = string(k-n,'0')+string(n,'1');
   do{
       space.push_back(onstate(s));
   }while(next_permutation(s.begin(), s.end()));
   return space;
}

onspace fock::get_fci_space(const int ks, const int na, const int nb){
   //cout << "\nfock::get_fci_space (ks,na,nb)=" << ks << "," << na << "," << nb << endl;
   onspace space_a = fock::get_fci_space(ks,na);
   onspace space_b = fock::get_fci_space(ks,nb);
   onspace space;
   for(size_t ia=0; ia<space_a.size(); ia++){
      for(size_t ib=0; ib<space_b.size(); ib++){
	 space.push_back( move(onstate(space_a[ia],space_b[ib])) );
      }
   }
   return space;
}

// coupling matrix: B0[b1,b] = <b0,b1|b>
matrix<double> fock::get_Bmatrix(const onstate& state0,
	 	                 const onspace& space1,
		                 const onspace& space){
   int m = space1.size(), n = space.size();
   matrix<double> B(m,n);
   for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
	 B(j,i) = static_cast<double>(state0.join(space1[j]) == space[i]);
      } // j
   } // i
   return B;
}
