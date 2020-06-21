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
 
   // for LiH3/sto-3g DHF [nfrozen=4]; compared against DIRAC/KRCI
   onspace space2 = get_fci_space(12,6);
   int dim = space2.size();

   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, "./cmole2.info");

   //int2e.set_real();
   int1e.set_real();
   ecore = 0.0;
   int2e.set_zero();
   //int1e.set_zero();

   // full diag
   auto eHF = get_Hii(space2[0], int2e, int1e) + ecore;
   cout << "ref=" << space2[0] << " eHF=" << setprecision(12) << eHF << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   cout << H.rows() << " diff=" << symmetric_diff(H) << endl;
   vector<double> e(H.rows());
   auto v(H);
   eig_solver(H, e, v); // Hc=ce
   cout << "e0=" << setprecision(12) << e[0] << " e1=" << e[1] << endl;
   //assert(abs(e[0] + 6.766940567056) < thresh);

   int nroot = 5;
   vector<double> es(nroot), es1(nroot);
   linalg::matrix<DTYPE> vs(dim,nroot), vs1(dim,nroot);
   
/*
   // simpleci solver
   fock::ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0] - e[0]) < thresh); 
   assert(abs(es[1] - e[1]) < thresh);
*/

   // directci solver
   fci::ci_solver(es1, vs1, space2, int2e, int1e, ecore);
   cout << "finished" << endl;
   exit(1);
   assert(abs(es1[0] - e[0]) < thresh);
   assert(abs(es1[1] - e[1]) < thresh);
   
/*
   // check eigenvalues
   cout << "\nCheck difference:" << endl;
   cout << defaultfloat << setprecision(12);
   for(int i=0; i<nroot; i++){
      cout << "i=" << i 
	   << " e=" << es[i] << " " << es1[i] 
	   << " diff=" << es1[i]-es[i] << endl;
      assert(abs(es[i]-es1[i])<1.e-10);
   }
   double e0 = -98.67325222947;
   assert(abs(es[0]-e0) < thresh);
   // check eigenvectors
   cout << "|vs1-vs|=" << normF(vs1-vs) << endl;
   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(space2, v0);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << SvN  << endl;
   assert(abs(SvN-0.176260153867) < thresh);
*/

   return 0;
}
