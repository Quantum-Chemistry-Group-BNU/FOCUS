#ifndef ONSPACE_H
#define ONSPACE_H

#include <vector>
#include "onstate.h"
#include "integral.h"
#include "matrix.h"

namespace fock{

using onspace = std::vector<onstate>;
      
// print
void check_space(onspace& space);

// spinless case
onspace get_fci_space(const int k, const int n);

// k - number of spatial orbitals 
onspace get_fci_space(const int ks, const int na, const int nb);

// -- Hamiltonian related ---

// generate represenation of H in this space
linalg::matrix get_Ham(const onspace& space,
		       const integral::two_body& int2e,
		       const integral::one_body& int1e,
		       const double ecore);

// solve eigenvalue problem in this space
void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	 
	       const onspace& space, 
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

// Hdiag
std::vector<double> get_Hdiag(const onspace& space,
		              const integral::two_body& int2e,
		              const integral::one_body& int1e,
		              const double ecore);

// y=H*x
void get_Hx(double* y,
	    const double* x, 
	    const onspace& space,
	    const integral::two_body& int2e,
	    const integral::one_body& int1e,
	    const double ecore);

// --- Direct product space ---

// coupling matrix: B0[b1,b] = <b0,b1|b>
linalg::matrix get_Bmatrix(const fock::onstate& state0,
		           const onspace& space1,
			   const onspace& space);

} // fock

#endif
