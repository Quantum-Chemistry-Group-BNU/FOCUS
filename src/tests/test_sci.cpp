#include <iostream>
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include "../utils/sci.h"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_sci(){
   cout << global::line_separator << endl;	
   cout << "test_sci" << endl;
   cout << global::line_separator << endl;	
  
   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;

   int k, ne;
   //integral::read_fcidump(int2e, int1e, ecore, "FCIDUMP_fes");
   //k = 73*2; ne = 2; // fes
   //integral::read_fcidump(int2e, int1e, ecore, "FCIDUMP_c2");
   //k = 12*2; ne = 8; // c2
   integral::read_fcidump(int2e, int1e, ecore, "FCIDUMP_lih");
   k =  6*2; ne = 4; // lih
   onspace fci_space = get_fci_space(k/2,ne/2,ne/2);
   int dim = fci_space.size();
 
   // iterative algorithm
   int nroot = 1; 
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
  
   // do something here
   auto int1e_AA = int1e.get_AA();
   auto int2e_AAAA = int2e.get_AAAA();
   
   ci_solver(es, vs, fci_space, int2e_AAAA, int1e_AA, ecore);
   sci::ci_solver(es, vs, fci_space, int2e_AAAA, int1e_AA, ecore);
   exit(1);

   ci_solver(es, vs, fci_space, int2e, int1e, ecore);
   sci::ci_solver(es, vs, fci_space, int2e_AAAA, int1e_AA, ecore);

   //sci::ci_solver(es, vs, fci_space, int2e, int1e, ecore);
   exit(1); 
   //ci_solver(es, vs, fci_space, int2e, int1e, ecore);
   
   // analysis 
   vector<double> v0i(vs.col(0),vs.col(0)+dim);
   fock::coefficients(fci_space, v0i);

   return 0;
}
