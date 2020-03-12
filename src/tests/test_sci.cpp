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
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   fock::coefficients(fci_space, v0);
   vector<double> sigs(v0.size());
   transform(v0.cbegin(),v0.cend(),sigs.begin(),
             [](const double& x){return pow(x,2);}); // pi=|ci|^2
   auto SvN = vonNeumann_entropy(sigs);
   assert(abs(es[0]+75.48440859785963) < 1.e-8);
   assert(abs(SvN-0.7211959135921441) < 1.e-5);
   exit(1);

   // compute rdm1
   linalg::matrix rdm1(k,k);
   fock::get_rdm1(fci_space,v0,v0,rdm1);
   cout << "tr(RDM1)=" << rdm1.trace() << endl;
   // compute rdm2
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
   fock::get_rdm2(fci_space,v0,v0,rdm2);
   // compute E
   cout << setprecision(12);
   cout << "e0=" << ecore << endl;
   double e1 = fock::get_e1(rdm1, int1e);
   cout << "e1=" << e1 << endl;
   double e2 = fock::get_e2(rdm2, int2e);
   cout << "e2=" << e2 << endl;
   auto etot = ecore+e1+e2;
   cout << "etot=" << etot << endl; 
   exit(1);

   return 0;
}
