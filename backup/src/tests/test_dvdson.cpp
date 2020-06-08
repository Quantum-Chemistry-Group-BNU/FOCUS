#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_dvdson(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_dvdson" << endl;
   cout << global::line_separator << endl;	
   
   double thresh = 1.e-6;

   // for LiH
   onspace space2 = get_fci_space(6,2,2);
   int dim = space2.size();
   check_space(space2);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, "../database/fcidump/FCIDUMP_lih");

   // iterative algorithm
   int nroot = 2; 
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0]+7.87388139034) < thresh); 
   assert(abs(es[1]+7.74509251524) < thresh);

   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(space2,v0);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << SvN  << endl;
   assert(abs(SvN-0.1834419989) < thresh);

   return 0;
}
