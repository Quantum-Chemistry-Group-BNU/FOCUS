#ifndef FCI_RDM_H
#define FCI_RDM_H

#include "../core/matrix.h"
#include "../core/onspace.h"
#include <vector>

namespace fci{

// <Psi1|p^+q|Psi2> (NR case)
void get_rdm1(const fock::onspace& space,
 	      const std::vector<double>& civec1,
	      const std::vector<double>& civec2,
	      linalg::matrix& rdm1);

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1) using sparseH
// which contains the computed connection information  
void get_rdm2(const fock::onspace& space,
	      const sparse_hamiltonian& sparseH,
 	      const std::vector<double>& civec1,
	      const std::vector<double>& civec2,
	      linalg::matrix& rdm2);

} // fci

#endif
