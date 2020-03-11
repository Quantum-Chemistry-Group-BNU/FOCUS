#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../settings/global.h"
#include "../utils/sci.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_sci(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_sci" << endl;
   cout << global::line_separator << endl;	
   
   double thresh = 1.e-6;
  
   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;

   int k, ne;
   integral::read_fcidump(int2e, int1e, ecore, "../fcidump/FCIDUMP_c2");
   k = 12*2; ne = 8; 
   onspace fci_space = get_fci_space(k/2,ne/2,ne/2);
   int dim = fci_space.size();

   int nroot = 1;
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);

   sci::ci_solver(es, vs, fci_space, int2e, int1e, ecore);

   // analysis 
   vector<double> v0i(vs.col(0),vs.col(0)+dim);
   fock::coefficients(fci_space, v0i);
   vector<double> sigs(v0i.size());
   transform(v0i.cbegin(),v0i.cend(),sigs.begin(),
             [](const double& x){return pow(x,2);}); // pi=|ci|^2
   auto SvN = vonNeumann_entropy(sigs);
   assert(abs(es[0]+75.48440859785963) < thresh);
   assert(abs(SvN-0.7211959135921441) < thresh);
   exit(1);

   return 0;
}
