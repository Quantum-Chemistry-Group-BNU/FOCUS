#ifndef HAMILTONIAN_H
#define HAMILTONIAN_H

#include "onstate.h"
#include "integral.h"

namespace fock{

double get_Hii(const onstate& state1,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

double get_HijS(const onstate& state1, const onstate& state2,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
		const int iop);

double get_HijD(const onstate& state1, const onstate& state2,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
		const int iop);

double get_Hij(const onstate& state1, const onstate& state2,
	       const integral::two_body& int2e,
	       const integral::one_body& int1e);

}

#endif
