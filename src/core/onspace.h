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
// coupling matrix for basis: B0[j,i] = <D[0],D[j]|D[i]>
template <typename Tm>
linalg::matrix<Tm> get_Bmatrix(const fock::onstate& state0,
		               const onspace& space1,
			       const onspace& space){
   int m = space1.size(), n = space.size();
   linalg::matrix<Tm> B(m,n);
   for(int i=0; i<n; i++){
      for(int j=0; j<m; j++){
	 B(j,i) = static_cast<double>(state0.join(space1[j]) == space[i]);
      } // j
   } // i
   return B;
}

} // fock

#endif
