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
      auto state = onstate(sub);
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
