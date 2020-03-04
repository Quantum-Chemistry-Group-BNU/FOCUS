#ifndef ONSPACE_H
#define ONSPACE_H

#include <iostream>
#include <vector>
#include <memory>
#include "onstate.h"
#include "integral.h"
#include "matrix.h"

namespace fock{

using onspace = std::vector<onstate>;
      
// print
void check_space(onspace& space);

// spinless case
onspace fci_space(const int k, const int n);

// k - number of spatial orbitals 
onspace fci_space(const int ks, const int na, const int nb);
      
// generate represenation of H in this space
linalg::matrix get_Ham(const onspace& space,
		       const integral::two_body& int2e,
		       const integral::one_body& int1e,
		       const double ecore=0.0);

// solve eigenvalue problem in this space
void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	 
	       const onspace& space, 
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore=0.0);

// Hdiag
std::vector<double> get_Hdiag(const onspace& space,
		              const integral::two_body& int2e,
		              const integral::one_body& int1e,
		              const double ecore);

// y = H*x
void get_Hx(double* y,
	    const double* x, 
	    const onspace& space,
	    const integral::two_body& int2e,
	    const integral::one_body& int1e,
	    const double ecore);

}

#endif
