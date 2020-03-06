#ifndef SCI_H
#define SCI_H

#include <vector>
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/onspace.h"

namespace sci{

void ci_solver(std::vector<double>& es,
	       linalg::matrix& vs,	
	       const fock::onspace& space,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

} // sci

#endif
