#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/hamiltonian.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

int tests::test_hamiltonian(){
   cout << endl;
   cout << global::line_separator << endl;	
   cout << "tests::test_hamiltonian" << endl;
   cout << global::line_separator << endl;	
   
   double thresh = 1.e-8;

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, "../database/fcidump/FCIDUMP_lih");

   // just for LiH
   onspace space1 = get_fci_space(4,2);
   onspace space2 = get_fci_space(6,2,2);

   // test get_Hij
   cout << space2[16] << endl;
   cout << space2[1] << endl;
   assert(space2[16].to_string() == "000000110011");
   assert(space2[1].to_string() == "000000100111");
   auto Ha = fock::get_Hij(space2[16], space2[1], int2e, int1e);
   auto Hb = fock::get_Hij(space2[1], space2[16], int2e, int1e);
   cout << setprecision(12);
   cout << Ha << endl;
   cout << Hb << endl;
   assert(abs(Ha-0.04854627932) < thresh);
   assert(abs(Hb-0.04854627932) < thresh);

   // build H and check symmetry
   cout << "\nCheck Hamiltonian & eigenvalue problem" << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   auto t0 = global::get_time();
   H = H.transpose();
   cout << setprecision(12);
   int ndiff = 0;
   for(int j=0; j<H.cols(); j++){
      for(int i=0; i<H.rows(); i++){
         if(abs(H(i,j))<thresh) continue; // skip small terms
	 if(abs(H(i,j)-H(j,i))<thresh) continue;
	 ndiff += 1; 
         cout << space2[i].diff_num(space2[j]) 
	      << " (" << i << "," << j << ")=" 
	      << H(i,j) << " " << H(j,i)
              << " diff=" << abs(H(i,j)-H(j,i))	   
              << endl;
      }
   }
   cout << "ndiff=" << ndiff << endl;
   assert(ndiff == 0);

   // full diagonalization
   linalg::matrix v(H);
   vector<double> e(H.rows());
   eigen_solver(v,e); // Hc=ce
   cout << "H: symmetric_diff=" << setprecision(12) 
        << symmetric_diff(H) << endl;  
   cout << "eigenvalues:\n" 
	<< e[0] << "\n" << e[1] << "\n" << e[2] << "\n"
	<< e[3] << "\n" << e[4] << "\n" << e[5] << endl;
   auto t1 = global::get_time();
   cout << "timing : " << setw(10) << fixed << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   // compared with FCI value
   assert(abs(e[0]+7.87388139034) < thresh); 
   assert(abs(e[1]+7.74509251524) < thresh);
   assert(abs(e[2]+7.72987790743) < thresh);
   assert(abs(e[3]+7.70051892907) < thresh);
   assert(abs(e[4]+7.70051892907) < thresh);
   assert(abs(e[5]+7.67557444095) < thresh);
   assert(symmetric_diff(H) < thresh);

   // check coefficient
   vector<double> v0(v.data(),v.data()+v.rows());
   coeff_population(space2,v0);
   auto SvN = coeff_entropy(v0);
   cout << "SvN=" << SvN  << endl;
   assert(abs(SvN-0.1834419989) < thresh);

   // compute rdm1
   int k = space2[0].size();
   linalg::matrix rdm1(k,k);
   fock::get_rdm1(space2,v0,v0,rdm1);
   rdm1.print("rdm1");
   cout << "tr(RDM1)=" << rdm1.trace() << endl;
   auto diag = rdm1.diagonal();
   for(int i=0; i<diag.size(); i++){
      cout << "i=" << i << " ni=" << diag[i] << endl;
   }
   
   // compute rdm2
   int k2 = k*(k-1)/2;
   linalg::matrix rdm2(k2,k2);
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
   
   return 0;
}
