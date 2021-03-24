#include <iostream>
#include <iomanip>
#include "tools.h"
#include "onspace.h"
#include "integral.h"
#include "hamiltonian.h"
#include "matrix.h"
#include "linalg.h"
#include "tests_core.h"

using namespace std;
using namespace fock;

int tests::test_hamiltonian(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_hamiltonian" << endl;
   cout << tools::line_separator << endl;	
   
   double thresh = 1.e-8;

   // read integral
   integral::two_body<complex<double>> int2e;
   integral::one_body<complex<double>> int1e;
   double ecore;
   
   // just for LiH
   integral::load(int2e, int1e, ecore, "./fmole.info");
   onspace space2 = get_fci_space(6,2,2);

   // test get_Hij
   cout << space2[16] << endl;
   cout << space2[1] << endl;
   assert(space2[16].to_string() == "000000110011");
   assert(space2[1].to_string() == "000000100111");
   auto Ha = fock::get_Hij(space2[16], space2[1], int2e, int1e);
   auto Hb = fock::get_Hij(space2[1], space2[16], int2e, int1e);
   cout << setprecision(12);
   cout << Ha << " " << std::abs(Ha-0.04854627932) << endl;
   cout << Hb << " " << std::abs(Hb-0.04854627932) << endl;
   assert(std::abs(Ha-0.04854627932) < thresh);
   assert(std::abs(Hb-0.04854627932) < thresh);

   // build H and check symmetry
   cout << "\nCheck Hamiltonian & eigenvalue problem" << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   auto t0 = tools::get_time();
   cout << setprecision(12);
   int ndiff = 0;
   for(int j=0; j<H.cols(); j++){
      for(int i=0; i<H.rows(); i++){
         if(std::abs(H(i,j))<thresh) continue; // skip small terms
	 if(std::abs(H(i,j)-H(j,i))<thresh) continue;
	 ndiff += 1; 
         cout << space2[i].diff_num(space2[j]) 
	      << " (" << i << "," << j << ")=" 
	      << H(i,j) << " " << H(j,i)
              << " diff=" << std::abs(H(i,j)-H(j,i))	   
              << endl;
      }
   }
   cout << "ndiff=" << ndiff << endl;
   assert(ndiff == 0);

   // full diagonalization
   vector<double> e(H.rows());
   auto v(H);
   eig_solver(H, e, v); // Hc=ce
   cout << "H: symmetric_diff=" << setprecision(12) 
        << symmetric_diff(H) << endl;  
   cout << "eigenvalues:\n" 
	<< e[0] << "\n" << e[1] << "\n" << e[2] << "\n"
	<< e[3] << "\n" << e[4] << "\n" << e[5] << endl;
   auto t1 = tools::get_time();
   cout << "timing : " << setw(10) << fixed << setprecision(2) 
	<< tools::get_duration(t1-t0) << " s" << endl;
   // compared with FCI value
   assert(std::abs(e[0]+7.87388139034) < thresh); 
   assert(std::abs(e[1]+7.74509251524) < thresh);
   assert(std::abs(e[2]+7.72987790743) < thresh);
   assert(std::abs(e[3]+7.70051892907) < thresh);
   assert(std::abs(e[4]+7.70051892907) < thresh);
   assert(std::abs(e[5]+7.67557444095) < thresh);
   assert(symmetric_diff(H) < thresh);

   return 0;
}
