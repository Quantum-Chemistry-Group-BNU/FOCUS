#include <iostream>
#include <iomanip>
#include <string>
#include "../io/input.h"
#include "ci_header.h"
#include "tests_ci.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_sci(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_sci" << endl;
   cout << tools::line_separator << endl;	

   // read input
   string fname = "input.dat";
   input::schedule schd;
   schd.read(fname);

   using DTYPE = complex<double>;
   
   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
  
   int nroots = schd.sci.nroots;
   vector<double> es(nroots,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<DTYPE>> vs(nroots);
   fci::sparse_hamiltonian<DTYPE> sparseH;
   sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);

   // pt2 for single root
   sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);

   return 0;
}
