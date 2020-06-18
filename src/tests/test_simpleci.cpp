#include "../core/tools.h"
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/matrix.h"
#include "../core/dvdson.h"
#include "../core/simpleci.h"
#include "../core/analysis.h"
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_simpleci(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_simpleci" << endl;
   cout << tools::line_separator << endl;	
   
   const double thresh = 1.e-6;

   // for LiH
   onspace space2 = get_fci_space(6,2,2);
   int dim = space2.size();

   // read integral
   integral::two_body<double> int2e;
   integral::one_body<double> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, "./fmole.info");

   // iterative algorithm
   int nroot = 3;
   vector<double> es(nroot);
   matrix<double> vs(dim,nroot);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   assert(abs(es[0]+7.87388139034) < thresh); 
   assert(abs(es[1]+7.74509251524) < thresh);
   vs.print("vs");

   // analysis 
   vector<double> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(space2,v0);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << SvN  << endl;
   assert(abs(SvN-0.1834419989) < thresh);

/*
   // compute rdm1
   int k = space2[0].size();
   matrix rdm1(k,k);
   fock::get_rdm1(space2,v0,v0,rdm1);
   rdm1.print("rdm1");
   cout << "tr(RDM1)=" << rdm1.trace() << endl;
   auto diag = rdm1.diagonal();
   for(int i=0; i<diag.size(); i++){
      cout << "i=" << i << " ni=" << diag[i] << endl;
   }
   
   // compute rdm2
   int k2 = k*(k-1)/2;
   matrix rdm2(k2,k2);
   fock::get_rdm2(space2,v0,v0,rdm2);
   rdm2.print("rdm2");
   cout << setprecision(12);
   cout << rdm2(0,0) << endl;
   assert(abs(rdm2(0,0)-0.999930449) < thresh);
   // check AAAA part
   for(int p0=0; p0<k; p0++){
      for(int p1=0; p1<p0; p1++){
         for(int q0=0; q0<k; q0++){
            for(int q1=0; q1<q0; q1++){
	       auto p01 = tools::canonical_pair0(p0,p1);
	       auto q01 = tools::canonical_pair0(q0,q1);
	       // AAAA-block
	       if(p0%2 == 0 && p1%2 == 0 && 
		  q0%2 == 0 && q1%2 == 0 &&
		  abs(rdm2(p01,q01))>1.e-5){
		  cout << "(p0,p1,q1,q0)=" 
		       << defaultfloat << setprecision(8)
		       << p0/2 << " "
		       << p1/2 << " "
		       << q1/2 << " "
		       << q0/2 << " "
		       << rdm2(p01,q01) << endl; 
	       }
	    }
	 }
      }
   }

   // compute E
   cout << setprecision(12);
   cout << "e0=" << ecore << endl;
   double e1 = fock::get_e1(rdm1, int1e);
   cout << "e1=" << e1 << endl;
   double e2 = fock::get_e2(rdm2, int2e);
   cout << "e2=" << e2 << endl;
   auto etot = ecore+e1+e2;
   cout << "etot=" << etot << endl; 
   assert(abs(ecore - 1.1824308303) < thresh);
   assert(abs(e1 - (-12.7502341297)) < thresh);
   assert(abs(e2 - 3.6939219091) < thresh);
   assert(abs(etot - (-7.87388139034)) < thresh);
*/   

   return 0;
}
