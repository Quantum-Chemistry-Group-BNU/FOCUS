#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <vector>
#include "onspace.h"

namespace fock{

void coefficients(const onspace& space, 
		  const std::vector<double>& civec, 
		  const double thresh=1.e-2);

double vonNeumann_entropy(const std::vector<double>& sigs, 
			  const double cutoff=1.e-12);

}

#endif
