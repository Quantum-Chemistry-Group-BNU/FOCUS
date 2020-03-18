#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <vector>
#include "onspace.h"
#include "matrix.h"
#include "integral.h"

namespace fock{

void coefficients(const onspace& space, 
		  const std::vector<double>& civec, 
		  const double thresh=1.e-2);

double vonNeumann_entropy(const std::vector<double>& sigs, 
			  const double cutoff=1.e-12);

// <Psi1|p^+q|Psi2> - this also allows to compute transition rdm
void get_rdm1(const onspace& space,
              const std::vector<double>& civec1,
              const std::vector<double>& civec2,
              linalg::matrix& rdm1);

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1)
void get_rdm2(const onspace& space,
  	      const std::vector<double>& civec1,
	      const std::vector<double>& civec2,
	      linalg::matrix& rdm2);

// from rdm2 for particle number conserving wf
linalg::matrix get_rdm1_from_rdm2(const linalg::matrix& rdm2);

// E1
double get_e1(const linalg::matrix& rdm1,
	      const integral::one_body& int1e);

// E2
double get_e2(const linalg::matrix& rdm2,
	      const integral::two_body& int2e);

// Etot
double get_etot(const linalg::matrix& rdm1,
		const linalg::matrix& rdm2,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const double ecore);

double get_etot(const linalg::matrix& rdm2,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const double ecore);

} // fock

#endif
