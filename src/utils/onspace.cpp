#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>
#include "../settings/global.h"
#include "onspace.h"
#include "hamiltonian.h"
#include "dvdson.h"

using namespace std;
using namespace fock;
using namespace linalg;

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
   cout << "\nfock::fci_space" << endl; 
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
   cout << "\nfock::fci_space" << endl; 
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
matrix fock::get_Ham(const onspace& space,
		     const integral::two_body& int2e,
		     const integral::one_body& int1e,
		     const double ecore){
   cout << "\nfock::fci_getHam" << endl; 
   auto dim = space.size();
   matrix H(dim,dim);
   // column major
   for(size_t j=0; j<dim; j++){
      for(size_t i=0; i<dim; i++){
         H(i,j) = get_Hij(space[i],space[j],int2e,int1e);
      }
      H(j,j) += ecore;
   }
   return H;
}

// Hdiag: generate diagonal of H in this space
vector<double> fock::get_Hdiag(const onspace& space,
			       const integral::two_body& int2e,
			       const integral::one_body& int1e,
			       const double ecore){
   cout << "\nfock::fci_getHdiag" << endl;
   auto dim = space.size();
   vector<double> diag(dim);
   for(size_t i=0; i<dim; i++){
      diag[i] = get_Hii(space[i],int2e,int1e) + ecore;
   }
   return diag;
}

// y = H*x
void fock::get_Hx(double* y,
		  const double* x,
		  const onspace& space,
		  const integral::two_body& int2e,
		  const integral::one_body& int1e,
		  const double ecore){
   // y[i] = sum_j H[i,j]*x[j] 
   auto dim = space.size();
   for(size_t i=0; i<dim; i++){
      y[i] = 0.0;
      for(size_t j=0; j<dim; j++){
	 auto Hij = get_Hij(space[i],space[j],int2e,int1e);
         y[i] += Hij*x[j]; 
      }
      y[i] += ecore*x[i];
   }
}

// solve eigenvalue problem in this space
void fock::ci_solver(vector<double>& es,
	       	     matrix& vs,	
		     const onspace& space,
	       	     const integral::two_body& int2e,
	       	     const integral::one_body& int1e,
	       	     const double ecore){
   cout << "\nfock::ci_solver" << endl; 
   auto t0 = chrono::high_resolution_clock::now();
   // Davidson solver 
   dvdsonSolver solver;
   solver.ndim = space.size();
   solver.neig = es.size();
   // Hdiag
   auto Diag = get_Hdiag(space, int2e, int1e, ecore);
   solver.Diag = Diag.data(); 
   // y=H*x, see https://en.cppreference.com/w/cpp/utility/functional/ref
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(fock::get_Hx, _1, _2, cref(space), cref(int2e), cref(int1e), ecore);
   // solve
   solver.solve_iter(es.data(), vs.data());
   //solver.full_diag(es.data(), vs.data());
   auto t1 = chrono::high_resolution_clock::now();
   cout << "timing : " << setw(10) << fixed << setprecision(2) 
	<< chrono::duration_cast<chrono::milliseconds>(t1-t0).count()*0.001 << " s" << endl;
}
