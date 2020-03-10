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
   integral::read_fcidump(int2e, int1e, ecore, "FCIDUMP_lih");

   // iterative algorithm
   int nroot = 2; 
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0]+7.87388139034) < thresh); 
   assert(abs(es[1]+7.74509251524) < thresh);

   // analysis 
   vector<double> v0i(vs.col(0),vs.col(0)+dim);
   fock::coefficients(space2,v0i);
   vector<double> sigs(v0i.size());
   transform(v0i.cbegin(),v0i.cend(),sigs.begin(),
	     [](const double& x){return pow(x,2);}); // pi=|ci|^2
   auto SvN = vonNeumann_entropy(sigs);
   cout << "p0=" << sigs[0] << endl;
   cout << "SvN=" << SvN  << endl;
   assert(abs(sigs[0]-0.9805968962) < thresh);
   assert(abs(SvN-0.1834419989) < thresh);

   return 0;
}
