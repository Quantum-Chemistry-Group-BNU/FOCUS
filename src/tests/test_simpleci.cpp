#include "../settings/global.h"
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"
#include "../core/simpleci.h"
//#include "../core/analysis.h"
#include <iostream>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_simpleci(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_simpleci" << endl;
   cout << global::line_separator << endl;	
   
   const double thresh = 1.e-6;

   // for LiH
   onspace space2 = get_fci_space(6,2,2);
   int dim = space2.size();

   // read integral
   integral::two_body<complex<double>> int2e;
   integral::one_body<complex<double>> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, "./fmole.info");

   // iterative algorithm
   int nroot = 3;
   vector<double> es(nroot);
   linalg::matrix<complex<double>> vs(dim,nroot);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0]+7.87388139034) < thresh); 
   assert(abs(es[1]+7.74509251524) < thresh);
   vs.print("vs");

/*
   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(space2,v0);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << SvN  << endl;
   assert(abs(SvN-0.1834419989) < thresh);
*/

   return 0;
}
