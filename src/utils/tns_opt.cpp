#include "tns_opt.h"
#include "tns_oper.h"

using namespace std;
using namespace tns;

// sweep optimizations for Comb
// see my former implementation of DMRG

void tns::opt_sweep(const input::schedule& schd,
	            comb& icomb,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
	            const double ecore){
   cout << "\ntns::opt_sweep" << endl;
   // creat environmental operators 
   oper_env_right(icomb, icomb, int2e, int1e, schd.scratch);
}

// H*x

// decimation

