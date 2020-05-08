#ifndef TNS_OPT_H
#define TNS_OPT_H

#include "../io/input.h"
#include "../core/integral.h"
#include "tns_comb.h"

namespace tns{

void opt_sweep(const input::schedule& schd,
	       comb& icomb, // initial comb wavefunction
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

} // tns

#endif
