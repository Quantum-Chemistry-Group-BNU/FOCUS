#ifndef SIMPLECI_H
#define SIMPLECI_H

#include <functional>
#include <iostream>
#include <vector>
#include "onspace.h"
#include "integral.h"
#include "matrix.h"
#include "hamiltonian.h"
#include "dvdson.h"
#include "tools.h"

namespace fock{

// Hdiag: generate diagonal of H in this space
template <typename Tm>
std::vector<double> get_Hdiag(const onspace& space,
		              const integral::two_body<Tm>& int2e,
		              const integral::one_body<Tm>& int1e,
		              const double ecore){
   std::cout << "\nfock::get_Hdiag" << std::endl;
   auto dim = space.size();
   std::vector<double> diag(dim);
   for(size_t i=0; i<dim; i++){
      diag[i] = get_Hii(space[i], int2e, int1e) + ecore;
   }
   return diag;
}

// Brute-force construction of y = H*x
template <typename Tm>
void get_Hx(Tm* y,
	    const Tm* x, 
	    const onspace& space,
	    const integral::two_body<Tm>& int2e,
	    const integral::one_body<Tm>& int1e,
	    const double ecore){
   // y[i] = sum_j H[i,j]*x[j] 
   size_t dim = space.size();
   for(size_t i=0; i<dim; i++){
      y[i] = 0.0;
      for(size_t j=0; j<dim; j++){
	 y[i] += get_Hij(space[i], space[j], int2e, int1e)*x[j];
      }
      y[i] += ecore*x[i];
   }
}

// solve eigenvalue problem in this space via Brute-force construction of H*x,
// which works best for small configuration space
template <typename Tm>
void ci_solver(std::vector<double>& es,
	       linalg::matrix<Tm>& vs,	 
	       const onspace& space, 
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore){
   std::cout << "\nfock::ci_solver dim=" << space.size() << std::endl; 
   auto t0 = tools::get_time();
   // Davidson solver 
   linalg::dvdsonSolver<Tm> solver;
   solver.ndim = space.size();
   solver.neig = es.size();
   // Hdiag
   auto Diag = get_Hdiag(space, int2e, int1e, ecore);
   solver.Diag = Diag.data(); 
   // y=H*x, see https://en.cppreference.com/w/cpp/utility/functional/ref
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(&fock::get_Hx<Tm>, _1, _2, 
		      cref(space), cref(int2e), cref(int1e), 
		      ecore); 
   // solve
   solver.solve_iter(es.data(), vs.data());
   //solver.solve_diag(es.data(), vs.data());
   auto t1 = tools::get_time();
   std::cout << "timing for fock::ci_solver : " << std::setprecision(2) 
	     << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // fock

#endif
