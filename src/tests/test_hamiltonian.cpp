#include <iostream>
#include "../utils/onstate.h"
#include "../utils/onspace.h"
#include "../utils/integral.h"
#include "../utils/hamiltonian.h"
#include "../utils/analysis.h"
#include "../utils/matrix.h"
#include "../utils/linalg.h"
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

   onspace space1 = fci_space(4,2);
   onspace space2 = fci_space(6,2,2);
   int dim = space2.size();
   cout << "dim=" << dim << endl; 
   /*
   for(const auto& s1 : space2){
      for(const auto& s2 : space2){
         cout << "s1,s2=" << s1 << " " << s2 << 
		 " diff=" << s1.num_diff(s2) << " " << s2.num_diff(s1) << endl;
      }
   }
   */

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_integral(int2e, int1e, ecore);

   cout << "\neigenvalue problem" << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   auto t0 = chrono::high_resolution_clock::now();
   linalg::matrix v(H);
   vector<double> e(H.rows());
   linalg::eig(v,e);
   cout << "eigenvalues:\n" << setprecision(10) 
	<< e[0] << "\n" << e[1] << "\n" << e[2] << endl;
   auto t1 = chrono::high_resolution_clock::now();
   cout << "timing : " << setw(10) << fixed << setprecision(2) 
	<< chrono::duration_cast<chrono::milliseconds>(t1-t0).count()*0.001 << " s" << endl;
   
   //vector<double> v0(&v(0,0),&v(0,0)+v.rows());
   vector<double> v0(v.data(),v.data()+v.rows());
   fock::coefficients(space2,v0);
   vector<double> sigs(v0.size());
   transform(v0.cbegin(),v0.cend(),sigs.begin(),
	     [](const double& x){return pow(x,2);});
   cout << "p0=" << sigs[0] << endl;
   cout << vonNeumann_entropy(sigs) << endl;

//   fock::get_rdm1(
   return 0;
}
