#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "onstate.h"
#include "integral.h"
#include <tuple>

namespace fock{

double get_Hii(const onstate& state1,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

std::pair<double,long> get_HijS(const onstate& state1, 
			        const onstate& state2,
	        	        const integral::two_body& int2e,
	        	        const integral::one_body& int1e);

std::pair<double,long> get_HijD(const onstate& state1, 
			        const onstate& state2,
	        	        const integral::two_body& int2e,
	        	        const integral::one_body& int1e);

double get_Hij(const onstate& state1, 
	       const onstate& state2,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

}

#endif
