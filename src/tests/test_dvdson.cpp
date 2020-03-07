#include <iostream>
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_dvdson(){
   cout << global::line_separator << endl;	
   cout << "test_dvdson" << endl;
   cout << global::line_separator << endl;	

   // for LiH
   onspace space2 = get_fci_space(6,2,2);
   int dim = space2.size();
   cout << "dim=" << dim << endl;
   check_space(space2);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_integral(int2e, int1e, ecore, "FCIDUMP_lih");

   // iterative algorithm
   int nroot = 1; 
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   
   // analysis 
   vector<double> v0i(vs.col(0),vs.col(0)+dim);
   fock::coefficients(space2,v0i);

   return 0;
}
