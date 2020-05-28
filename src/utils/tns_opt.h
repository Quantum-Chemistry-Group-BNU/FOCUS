#ifndef TNS_OPT_H
#define TNS_OPT_H

#include "../io/input.h"
#include "../core/integral.h"
#include "tns_comb.h"
#include "tns_oper.h"

namespace tns{

void opt_sweep(const input::schedule& schd,
	       comb& icomb, // initial comb wavefunction
	       const integral::two_body& int2e,
	       const integral::one_body& int1e,
	       const double ecore);

void opt_onedot(const input::schedule& schd,
	        comb& icomb, 
		directed_bond& dbond,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const double ecore,
   	        const int dcut,
		std::vector<double>& eopt,
		double& dwt,
		int& deff);

void opt_twodot(const input::schedule& schd,
	        comb& icomb, 
		directed_bond& dbond,
	        const integral::two_body& int2e,
	        const integral::one_body& int1e,
	        const double ecore,
   	        const int dcut,
		std::vector<double>& eopt,
		double& dwt,
		int& deff);

using tm = std::chrono::high_resolution_clock::time_point;
void opt_timing(const std::vector<tm> ts);

} // tns

#endif
