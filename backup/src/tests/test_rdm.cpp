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

int tests::test_rdm(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_rdm" << endl;
   cout << global::line_separator << endl;	
   
   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;

   int k, ne;
   //integral::read_fcidump(int2e, int1e, ecore, "../database/fcidump/FCIDUMP_lih");
   //k = 6*2; ne = 4; 
   integral::read_fcidump(int2e, int1e, ecore, "../database/fcidump/FCIDUMP_c2");
   k = 12*2; ne = 8; 
   onspace fci_space = get_fci_space(k/2,ne/2,ne/2);
   int dim = fci_space.size();

   int nroot = 1;
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
   fci::sparse_hamiltonian sparseH;
   fci::ci_solver(sparseH, es, vs, fci_space, int2e, int1e, ecore);
   if(k == 12*2 && ne == 8) assert(abs(es[0]+75.48440859785963) < 1.e-8);

   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(fci_space, v0);
   auto SvN = coeff_entropy(v0);
   if(k == 12*2 && ne == 8) assert(abs(SvN-0.7211959135921441) < 1.e-5);

   // make_rdm2 from sparseH
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
   fci::get_rdm2(sparseH,fci_space,v0,v0,rdm2);
   double etot = fock::get_etot(rdm2,int2e,int1e,ecore);
   cout << "etot(rdm)=" << setprecision(12) << etot << endl;
   assert(abs(etot-es[0]) < 1.e-8);

   // get_rdm1_from_rdm2
   auto rdm1 = fock::get_rdm1_from_rdm2(rdm2);
   // compute E
   double e1 = fock::get_e1(rdm1, int1e);
   double e2 = fock::get_e2(rdm2, int2e);
   etot = ecore+e1+e2;
   cout << setprecision(12);
   cout << "e0=" << ecore << endl;
   cout << "e1=" << e1 << endl;
   cout << "e2=" << e2 << endl;
   cout << "etot=" << etot << endl; 
   assert(abs(es[0]-etot) < 1.e-8);

   // compute rdm1
   linalg::matrix rdm1b(k,k);
   fci::get_rdm1(fci_space,v0,v0,rdm1b);
   auto diff = normF(rdm1b-rdm1);
   cout << "|rdm1b-rdm1|=" << diff << endl;
   assert(diff < 1.e-6);
  
   return 0;
}
