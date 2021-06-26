#include "tools.h"
#include "onspace.h"
#include "integral.h"
#include "matrix.h"
#include "dvdson.h"
#include "simpleci.h"
#include "analysis.h"
#include "simplerdm.h"
#include "tests_core.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_simpleci(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_simpleci" << endl;
   cout << tools::line_separator << endl;	
   
   const double thresh = 1.e-6;
   using DTYPE = complex<double>;

   // for HF/sto-3g DHF
   onspace space2 = get_fci_space(12,10);
   int dim = space2.size();

   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, "./cmole.info");

   // --- full diag ---
   auto eHF = get_Hii(space2[0], int2e, int1e) + ecore;
   cout << "ref=" << space2[0] << " eHF=" << setprecision(12) << eHF << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   cout << H.rows() << " diff=" << symmetric_diff(H) << endl;
   vector<double> e(H.rows());
   auto v(H);
   eig_solver(H, e, v); // Hc=ce
   cout << "e0=" << setprecision(12) << e[0] << " e1=" << e[1] << endl;
   assert(std::abs(e[0] + 98.67325222947) < thresh);
   assert(std::abs(e[1] + 98.31758870307) < thresh);

   // analysis 
   vector<DTYPE> v0x(v.col(0),v.col(0)+dim);
   coeff_population(space2, v0x, 1.e-3);
   auto SvNx = coeff_entropy(v0x);
   cout << "SvN=" << setprecision(12) << SvNx << endl;
   assert(std::abs(SvNx-0.176260153867) < thresh);

   // --- iterative algorithm --- 
   int nroots = 3;
   vector<double> es(nroots);
   matrix<DTYPE> vs(dim,nroots);
   ci_solver(es, vs, space2, int2e, int1e, ecore);
   cout << "e0=" << setprecision(12) << es[0] << " e1=" << es[1] << endl;
   assert(std::abs(es[0] - e[0]) < thresh); 
   assert(std::abs(es[1] - e[1]) < thresh);
   vs.print("vs");

   // analysis 
   vector<DTYPE> v0(vs.col(0),vs.col(0)+dim);
   coeff_population(space2, v0, 1.e-3);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << setprecision(12) << SvN  << endl;
   assert(std::abs(SvN-0.176260153867) < thresh);

   // --- rdm1 / rdm2 --- 
   // compute rdm1
   int k = space2[0].size();
   matrix<DTYPE> rdm1(k,k);
   get_rdm1(space2, v0, v0, rdm1);
   //rdm1.print("rdm1");
   cout << "tr(RDM1)=" << rdm1.trace() << endl;
   auto diag = rdm1.diagonal();
   for(int i=0; i<diag.size(); i++){
      cout << "i=" << i << " ni=" << diag[i] << endl;
   }
   cout << "diff=" << symmetric_diff(rdm1) << endl;
   
   // compute rdm2
   int k2 = k*(k-1)/2;
   matrix<DTYPE> rdm2(k2,k2);
   get_rdm2(space2,v0,v0,rdm2);
   //rdm2.print("rdm2");
   cout << setprecision(12);
   cout << rdm2(0,0) << endl;
   assert(std::abs(rdm2(0,0) - 0.9999994014) < thresh);
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
		  std::abs(rdm2(p01,q01))>1.e-5){
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
   cout << "diff=" << symmetric_diff(rdm2) << endl;

   auto rdm1b = get_rdm1_from_rdm2(rdm2);
   auto diff1 = normF(rdm1b-rdm1);
   cout << "diffRDM1=" << diff1 << endl;
   assert(diff1 < thresh);

   // compute E
   cout << setprecision(12);
   cout << "e0=" << ecore << endl;
   double e1 = get_e1(rdm1, int1e);
   cout << "e1=" << e1 << endl;
   double e2 = get_e2(rdm2, int2e);
   cout << "e2=" << e2 << endl;
   auto etot = get_etot(rdm2, rdm1, int2e, int1e, ecore);
   cout << "etot=" << etot << " e0=" << e[0] << " diff=" << e[0]-etot << endl; 
   assert(std::abs(ecore - 4.76259489828) < thresh);
   assert(std::abs(e1 - (-148.922894885)) < thresh);
   assert(std::abs(e2 - 45.4870477569) < thresh);
   assert(std::abs(etot - e[0]) < thresh);

   return 0;
}
