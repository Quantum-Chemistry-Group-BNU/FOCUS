#include "../core/tools.h"
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"
#include "../core/simpleci.h"
#include "../core/analysis.h"
#include "../sci/fci.h"
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_fci(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_fci" << endl;
   cout << tools::line_separator << endl;	
   
   const double thresh = 1.e-5;
   using DTYPE = complex<double>;
 
   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   
   // for LiH3/sto-3g DHF [nfrozen=4]; compared against DIRAC/KRCI
   integral::load(int2e, int1e, ecore, "./cmole2.info");
   onspace space2 = get_fci_space(int1e.sorb,6);
   
   int dim = space2.size();
   auto eHF = get_Hii(space2[0], int2e, int1e) + ecore;
   cout << "\nref=" << space2[0] << " eHF=" << setprecision(12) << eHF << endl;

   // full diag
   auto H = get_Ham(space2,int2e,int1e,ecore);
   cout << "diff=" << symmetric_diff(H) << endl;
   vector<double> e(H.rows());
   auto v(H);
   eig_solver(H, e, v); // Hc=ce
   cout << "e0=" << setprecision(12) << e[0] << " e1=" << e[1] << endl;
   assert(abs(e[0] + 6.766940567056) < thresh);

   int nroot = 1;
   vector<double> es(nroot), es1(nroot);
   linalg::matrix<DTYPE> vs(dim,nroot), vs1(dim,nroot);

/*
   // simpleci solver
   fock::ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0] - e[0]) < thresh); 
   assert(abs(es[nroot-1] - e[nroot-1]) < thresh);
*/

   // directci solver
   fci::ci_solver(es1, vs1, space2, int2e, int1e, ecore);
   assert(abs(es1[0] - e[0]) < thresh);
   assert(abs(es1[nroot-1] - e[nroot-1]) < thresh);

   return 0;
}
