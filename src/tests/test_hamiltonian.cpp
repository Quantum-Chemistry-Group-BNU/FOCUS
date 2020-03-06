#include <iostream>
#include "../core/onstate.h"
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/hamiltonian.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_hamiltonian(){
   cout << global::line_separator << endl;	
   cout << "test_hamiltonian" << endl;
   cout << global::line_separator << endl;	

   // just for LiH
   onspace space1 = get_fci_space(4,2);
   onspace space2 = get_fci_space(6,2,2);
   int dim = space2.size();
   cout << "dim=" << dim << endl; 

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_integral(int2e, int1e, ecore, "FCIDUMP_lih");

   cout << "\neigenvalue problem" << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   auto t0 = chrono::high_resolution_clock::now();
   linalg::matrix v(H);
   vector<double> e(H.rows());
   eigen_solver(v,e);
   cout << "eigenvalues:\n" << setprecision(10) 
	<< e[0] << "\n" << e[1] << "\n" << e[2] << endl;
   auto t1 = chrono::high_resolution_clock::now();
   cout << "timing : " << setw(10) << fixed << setprecision(2) 
	<< chrono::duration_cast<chrono::milliseconds>(t1-t0).count()*0.001 << " s" << endl;
   
   vector<double> v0(v.data(),v.data()+v.rows());
   fock::coefficients(space2,v0);
   vector<double> sigs(v0.size());
   // pi=|ci|^2
   transform(v0.cbegin(),v0.cend(),sigs.begin(),
	     [](const double& x){return pow(x,2);});
   cout << "p0=" << sigs[0] << endl;
   cout << vonNeumann_entropy(sigs) << endl;

   // compute rdm
   int k = space2[0].size();
   linalg::matrix rdm1(k,k);
   fock::get_rdm1(space2,v0,v0,rdm1);
   rdm1.print("rdm1");
   cout << "tr(RDM1)=" << rdm1.trace() << endl;
   auto diag = rdm1.diagonal();
   for(int i=0; i<diag.size(); i++){
      cout << "i=" << i << " ni=" << diag[i] << endl;
   }
   
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
   fock::get_rdm2(space2,v0,v0,rdm2);
   rdm2.print("rdm2");
   cout << rdm2(0,0) << endl;
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
   cout << "e0=" << ecore << endl;
   double e1 = fock::get_e1(rdm1, int1e);
   cout << "e1=" << e1 << endl;
   double e2 = fock::get_e2(rdm2, int2e);
   cout << "e2=" << e2 << endl;
   cout << "etot=" << ecore+e1+e2 << endl; 

   return 0;
}
