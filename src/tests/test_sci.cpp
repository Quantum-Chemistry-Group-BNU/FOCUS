#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../ci/fci_util.h"
#include "../ci/fci_rdm.h"
#include "../ci/sci.h"
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
   cout << tools::line_separator << endl;	
   cout << "tests::test_sci" << endl;
   cout << tools::line_separator << endl;	

   // read input
   input::schedule schd;
   input::read_input(schd);

   using DTYPE = complex<double>;
   
   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
  
   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<DTYPE>> vs(nroot);
   fci::sparse_hamiltonian<DTYPE> sparseH;
   sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);

   return 0;
}
