#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <chrono>
#include "onspace.h"
#include "hamiltonian.h"
#include "dvdson.h"
#include "../settings/global.h"

using namespace std;
using namespace fock;
using namespace linalg;

void fock::check_space(onspace& space){
   if(global::print_level>0){
      cout << "\nfock::check_space dim=" << space.size() << endl; 
      for(size_t i=0; i<space.size(); i++){
          cout << "i=" << i << " : " << space[i] << endl;
      }
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

// generate represenation of H in this space
matrix fock::get_Ham(const onspace& space,
		     const integral::two_body& int2e,
		     const integral::one_body& int1e,
		     const double ecore){
   cout << "\nfock::get_Ham" << endl; 
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
   cout << "\nfock::get_Hdiag" << endl;
   auto dim = space.size();
   vector<double> diag(dim);
   for(size_t i=0; i<dim; i++){
      diag[i] = get_Hii(space[i],int2e,int1e) + ecore;
   }
   return diag;
}

// Brute-force construction of y = H*x
void fock::get_Hx(double* y,
		  const double* x,
		  const onspace& space,
		  const integral::two_body& int2e,
		  const integral::one_body& int1e,
		  const double ecore){
   // y[i] = sum_j H[i,j]*x[j] 
   size_t dim = space.size();
   for(size_t i=0; i<dim; i++){
      y[i] = 0.0;
      for(size_t j=0; j<dim; j++){
	 y[i] += get_Hij(space[i],space[j],int2e,int1e)*x[j];
      }
      y[i] += ecore*x[i];
   }
}

// solve eigenvalue problem in this space via Brute-force construction of H*x,
// which works best for small configuration space
void fock::ci_solver(vector<double>& es,
	       	     matrix& vs,	
		     const onspace& space,
	       	     const integral::two_body& int2e,
	       	     const integral::one_body& int1e,
	       	     const double ecore){
   cout << "\nfock::ci_solver dim=" << space.size() << endl; 
   auto t0 = global::get_time();
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
   //solver.solve_diag(es.data(), vs.data());
   auto t1 = global::get_time();
   cout << "timing for fock::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
