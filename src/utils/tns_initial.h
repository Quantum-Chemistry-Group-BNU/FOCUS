#ifndef TNS_INITIAL_H
#define TNS_INITIAL_H

#include "tns_comb.h"

namespace tns{

std::vector<qtensor2> get_cwf0(const qtensor3& rsite0);

void initial_onedot(comb& icomb);

void initial_twodot(comb& icomb,
		    const directed_bond& dbond,
		    qtensor4& wf,
		    const int nsub,
		    const int neig,
		    std::vector<double>& v0);

} // tns

#endif
