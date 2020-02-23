#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include "onstate.h"
#include "onspace.h"
#include "../settings/global.h"

using namespace std;
using namespace fock;

void fock::check_space(onspace& space){
   if(global::print_level>0){
      cout << "\nfock::check_space" << endl; 
      cout << "dim=" << space.size() << endl;
      for(size_t i=0; i<space.size(); i++){
          cout << "i=" << i << " : " << space[i] << endl;
      }
   }
}

onspace fock::fci_space(const int k, const int n){
   assert(k >= n);
   onspace space;
   string s = string(k-n,'0')+string(n,'1');
   do{
       space.push_back(onstate(s));
   }while(next_permutation(s.begin(), s.end()));
   check_space(space);
   return space;
}

onspace fock::fci_space(const int k, const int na, const int nb){
   assert(k%2 == 0);
   onspace space_a = fock::fci_space(k/2,na);
   onspace space_b = fock::fci_space(k/2,nb);
   onspace space;
   for(size_t ia=0; ia<space_a.size(); ia++){
      for(size_t ib=0; ib<space_b.size(); ib++){
	 space.push_back( move(onstate(space_a[ia],space_b[ib])) );
      }
   }
   check_space(space);
   return space;
}
