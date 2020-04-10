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
#include "../utils/tns.h"
#include "../settings/global.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_order(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_order" << endl;
   cout << global::line_separator << endl;	

   // read input
   string fname = "input.dat";
   input::schedule schd;
   input::read_input(schd,fname);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, 
		   	  schd.integral_file,
		    	  schd.integral_type);

   // analysis of hij, Jij, Kij
   integral::save_text_sym1e(int1e.data,"hpq");
   integral::save_text_sym1e(int2e.J,"jpq");
   integral::save_text_sym1e(int2e.K,"kpq");

   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<double>> vs(nroot);
   fci::sparse_hamiltonian sparseH;
   sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
   
   coeff_population(sci_space, vs[0]);

   // tns
   vector<int> order;
   double Smin;
   tns::ordering_fiedler(int2e.K, order);
   tns::ordering_ga(sci_space, vs, order, Smin); 

   return 0;
}
