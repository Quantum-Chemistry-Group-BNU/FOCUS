#ifndef PT2_H
#define PT2_H

#include "sci.h"

namespace sci{

// for single state	
void pt2_solver(const input::schedule& schd,
	        const double e0,
	        const std::vector<double>& v0,
	        const fock::onspace& space,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const double ecore);

} // sci

#endif
