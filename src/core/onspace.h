#ifndef ONSPACE_H
#define ONSPACE_H

#include <vector>
#include "onstate.h"
#include "matrix.h"

namespace fock{

using onspace = std::vector<onstate>;
      
// print
void check_space(onspace& space);

// --- FCI space ---
// spinless case
onspace get_fci_space(const int k, const int n);
// k - number of spatial orbitals 
onspace get_fci_space(const int ks, const int na, const int nb);

// --- Direct product space ---
// coupling matrix: B0[b1,b] = <b0,b1|b>
linalg::matrix<double> get_Bmatrix(const fock::onstate& state0,
		           	   const onspace& space1,
			   	   const onspace& space);

} // fock

#endif
