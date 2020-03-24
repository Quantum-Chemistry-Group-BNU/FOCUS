#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../utils/fci.h"
#include "../utils/fci_rdm.h"
#include "../utils/sci.h"
#include "../settings/global.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_sci(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_sci" << endl;
   cout << global::line_separator << endl;	

   // read input
   string fname = "input.dat";
   input::schedule schd;
   input::read_input(schd,fname);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, schd.integral_file);
  
   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<double>> vs(nroot);
   fci::sparse_hamiltonian sparseH;
   sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);

   // analysis 
   auto v0 = vs[0];
   coeff_population(sci_space, v0);
   auto SvN = coeff_entropy(v0);

   // compute rdm1
   int k = int1e.sorb; 
   linalg::matrix rdm1(k,k);
   fci::get_rdm1(sci_space,v0,v0,rdm1);

   // natural orbitals
   linalg::matrix u;
   vector<double> occ;
   fci::get_natural_nr(rdm1,u,occ);

/*
   // compute rdm2
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
   fci::get_rdm2(sci_space,sparseH,v0,v0,rdm2);
   exit(1);

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
   //if(k == 12*2 && ne == 8){
   //   assert(abs(es[0]+75.48440859785963) < 1.e-8);
   //   assert(abs(SvN-0.7211959135921441) < 1.e-5);
   //}
*/
   exit(1);

   return 0;
}
