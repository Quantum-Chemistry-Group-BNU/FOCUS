#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include "../settings/global.h"
#include "onspace.h"
#include "hamiltonian.h"

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

onspace fock::fci_space(const int ks, const int na, const int nb){
   onspace space_a = fock::fci_space(ks,na);
   onspace space_b = fock::fci_space(ks,nb);
   onspace space;
   for(size_t ia=0; ia<space_a.size(); ia++){
      for(size_t ib=0; ib<space_b.size(); ib++){
	 space.push_back( move(onstate(space_a[ia],space_b[ib])) );
      }
   }
   check_space(space);
   return space;
}

// generate represenation of H in this space
unique_ptr<double[]> fock::get_Ham(const onspace& space,
		                   const integral::two_body& int2e,
			           const integral::one_body& int1e,
			           const double ecore)
{
   auto dim = space.size();
   unique_ptr<double[]> H(new double[dim*dim]);
   // row major - H[i,j] (col major would be H[j*dim+i]=Hij (transposed)
   for(size_t i=0; i<dim; i++){
      for(size_t j=0; j<dim; j++){
         H[i*dim+j] = get_Hij(space[i],space[j],int2e,int1e);
      }
      H[i*dim+i] += ecore;
   }
   return H;
}
