#ifndef TNS_INITIAL_H
#define TNS_INITIAL_H

#include "tns_comb.h"

namespace tns{

std::vector<qtensor2> get_cwf0(const qtensor3& rsite0);

void initial_onedot(const comb& icomb,
		    const int nsub,
		    const int neig,
		    std::vector<double>& v0);

} // tns

#endif
