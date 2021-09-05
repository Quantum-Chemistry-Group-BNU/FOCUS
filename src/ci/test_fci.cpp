#include "../core/tools.h"
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"
#include "../core/simpleci.h"
#include "../core/analysis.h"
#include "ci_header.h"
#include "tests_ci.h"

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
   auto H = get_Hmat(space2,int2e,int1e,ecore);
   cout << "diff(H-H.h)=" << symmetric_diff(H) << endl;
   vector<double> e(H.rows());
   auto v(H);
   eig_solver(H, e, v); // Hc=ce
   cout << "e0[FCI]=" << setprecision(12) << e[0] << endl; 
   cout << "e1[FCI]=" << setprecision(12) << e[1] << endl;
   assert(std::abs(e[0] + 6.766940567056) < thresh);
   assert(std::abs(e[1] + 6.68030079306 ) < thresh);

   int nroots = 1;
   vector<double> es(nroots), es1(nroots);
   linalg::matrix<DTYPE> vs(dim,nroots), vs1(dim,nroots);

/*
   // simpleci solver
   fock::ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(std::abs(es[0] - e[0]) < thresh); 
   assert(std::abs(es[nroots-1] - e[nroots-1]) < thresh);
*/

   // directci solver
   fci::sparse_hamiltonian<DTYPE> sparseH;
   fci::ci_solver(sparseH, es1, vs1, space2, int2e, int1e, ecore);
   assert(std::abs(es1[0] - e[0]) < thresh);
   assert(std::abs(es1[nroots-1] - e[nroots-1]) < thresh);

   // --- rdm --- 
   vector<DTYPE> v0(vs1.col(0),vs1.col(0)+dim);
   coeff_population(space2, v0);
   auto SvN = coeff_entropy(v0);
   // make_rdm2 from sparseH
   int k = int1e.sorb;
   int k2 = k*(k-1)/2;
   linalg::matrix<DTYPE> rdm2(k2,k2);
   fci::get_rdm2(sparseH,space2,v0,v0,rdm2);
   
   double etot = fock::get_etot(rdm2,int2e,int1e,ecore);
   cout << "etot(rdm)=" << setprecision(12) << etot << endl;
   assert(std::abs(etot-es1[0]) < 1.e-8);

   // compute rdm1
   auto rdm1 = fock::get_rdm1_from_rdm2(rdm2);
   linalg::matrix<DTYPE> rdm1b(k,k);
   fci::get_rdm1(space2,v0,v0,rdm1b);
   auto diff = normF(rdm1b-rdm1);
   cout << "|rdm1b-rdm1|=" << diff << endl;
   assert(diff < 1.e-6);

   return 0;
}
