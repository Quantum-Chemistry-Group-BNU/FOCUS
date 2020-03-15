#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../settings/global.h"
#include "../utils/fci.h"
#include "../utils/fci_rdm.h"
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
   //integral::read_fcidump(int2e, int1e, ecore, "../fcidump/FCIDUMP_lih");
   //k = 6*2; ne = 4; 
   integral::read_fcidump(int2e, int1e, ecore, "../fcidump/FCIDUMP_c2");
   k = 12*2; ne = 8; 
   onspace fci_space = get_fci_space(k/2,ne/2,ne/2+1);
   int dim = fci_space.size();

   int nroot = 1;
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);

   fci::ci_solver(es, vs, fci_space, int2e, int1e, ecore);
   exit(1);

   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   fock::coefficients(fci_space, v0);
   vector<double> sigs(v0.size());
   transform(v0.cbegin(),v0.cend(),sigs.begin(),
             [](const double& x){return pow(x,2);}); // pi=|ci|^2
   auto SvN = vonNeumann_entropy(sigs);

   // compute rdm1
   linalg::matrix rdm1(k,k);
   fci::get_rdm1(fci_space,v0,v0,rdm1);

   // compute rdm2
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
   fci::get_rdm2(fci_space,v0,v0,rdm2);

   // compute E
   cout << setprecision(12) << endl;
   cout << "e0=" << ecore << endl;
   double e1 = fock::get_e1(rdm1, int1e);
   cout << "e1=" << e1 << endl;
   double e2 = fock::get_e2(rdm2, int2e);
   cout << "e2=" << e2 << endl;
   auto etot = ecore+e1+e2;
   cout << "etot=" << etot << endl; 
   assert(abs(es[0]-etot) < 1.e-8);

   // get_rdm1_from_rdm2
   auto rdm1b = fock::get_rdm1_from_rdm2(rdm2);
   auto diff = normF(rdm1b-rdm1);
   cout << "|rdm1b-rdm1|=" << diff << endl;
   assert(diff < 1.e-6);
  
   // get_etot 
   double etot1 = fock::get_etot(rdm1,rdm2,int2e,int1e,ecore);
   cout << "etot1=" << etot1 << endl;
   assert(abs(etot1-etot) < 1.e-8);

   // check for FCIDUMP_c2
   if(k == 12*2 && ne == 8){
      assert(abs(es[0]+75.48440859785963) < 1.e-8);
      assert(abs(SvN-0.7211959135921441) < 1.e-5);
   }
   exit(1);

   return 0;
}
