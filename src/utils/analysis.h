#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <vector>
#include "onspace.h"
#include "matrix.h"

namespace fock{

void coefficients(const onspace& space, 
		  const std::vector<double>& civec, 
		  const double thresh=1.e-2);

double vonNeumann_entropy(const std::vector<double>& sigs, 
			  const double cutoff=1.e-12);

void get_rdm1(const onspace& space,
	      const vector<double>& civec1,
	      const vector<double>& civec2,
	      linalg::matrix& rdm1);

void get_rdm2(const onspace& space,
	      const vector<double>& civec1,
	      const vector<double>& civec2,
	      linalg::matrix& rdm2);

}

#endif
