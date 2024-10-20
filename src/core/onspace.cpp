#include <iostream>
#include <cassert>
#include <string>
#include <algorithm>
#include <functional>
#include <bitset>
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

onspace fock::get_fci_space(const int k){
   const int kmax = 64; // [in fact it only works for much smaller k!]
   assert(k <= kmax);
   onspace space;
   for(size_t i=0; i<std::pow(2,k); i++){
      std::string s = std::bitset<kmax>(i).to_string(); // string conversion
      auto sub = s.substr(kmax-k,kmax); // 00000a
      auto state = onstate(sub,1);
      space.push_back(state);
   }
   return space;
}

onspace fock::get_fci_space(const int k, const int n){
   //cout << "\nfock::get_fci_space (k,n)=" << k << "," << n << endl; 
   assert(k >= n);
   onspace space;
   string s = string(k-n,'0')+string(n,'1');
   do{
      space.push_back(onstate(s,1));
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

// k - no. of orbitals, n - no. of alpha electrons
onspace fock::get_fci_space_single(const int k, const int n){
   assert(k >= n);
   onspace space;
   string s = string(k-n,'a')+string(n,'b');
   do{
      // flip is added to be consistent with ordering in the previous function
      space.push_back( onstate(s).flip() ); 
   }while(next_permutation(s.begin(), s.end()));
   return space;
}

onspace fock::convert_space(const unsigned long *bra, const int sorb, const int n){
   onspace space;
   int _len = (sorb-1)/64+1;
   for(size_t i = 0; i < n; i++){
      space.push_back(onstate(&bra[i*_len], sorb));
   }
   return space;
}
