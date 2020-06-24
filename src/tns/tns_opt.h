#ifndef TNS_OPT_H
#define TNS_OPT_H

#include "../io/input.h"
#include "../core/integral.h"
#include "tns_comb.h"
#include "tns_oper.h"

namespace tns{

void opt_sweep(const input::schedule& schd,
	       comb& icomb, // initial comb wavefunction
	       const integral::two_body<double>& int2e,
	       const integral::one_body<double>& int1e,
	       const double ecore);

void opt_finaldot(const input::schedule& schd,
		  comb& icomb,
	          const integral::two_body<double>& int2e,
	          const integral::one_body<double>& int1e,
		  const double ecore);

void opt_onedot(const input::schedule& schd,
   	        const input::sweep_ctrl& ctrl,
	        comb& icomb, 
		const directed_bond& dbond,
	        const integral::two_body<double>& int2e,
	        const integral::one_body<double>& int1e,
	        const double ecore,
		std::vector<double>& eopt,
		double& dwt,
		int& deff);

void opt_twodot(const input::schedule& schd,
   	        const input::sweep_ctrl& ctrl,
	        comb& icomb, 
		const directed_bond& dbond,
	        const integral::two_body<double>& int2e,
	        const integral::one_body<double>& int1e,
	        const double ecore,
		std::vector<double>& eopt,
		double& dwt,
		int& deff);

using tm = std::chrono::high_resolution_clock::time_point;
void opt_timing_analysis(const std::vector<tm> ts);

void opt_sweep_print(const input::schedule& schd,
		     const int isweep,
		     const comb& icomb,
		     const std::vector<directed_bond>& sweeps,
		     std::vector<std::pair<double,std::vector<double>>>& sweep_data,
		     const std::vector<double>& timing,
		     const std::vector<std::vector<double>>& eopt,
		     const std::vector<double>& dwt,
		     const std::vector<int>& deff);

} // tns

#endif
