#include <iostream>
#include "../utils/onstate.h"
#include "../utils/onspace.h"
#include "../utils/integral.h"
#include "../utils/hamiltonian.h"
#include "../utils/analysis.h"
#include "../settings/global.h"
#include <Eigen/Dense>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace Eigen;

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

   // Eigen3
   cout << "\neigenvalue problem" << endl;
   auto H = get_Ham(space2,int2e,int1e,ecore);
   Map<MatrixXd> Hm(H.get(),dim,dim);
   //cout << "Hm\n" << Hm << endl;
   
   auto time0 = chrono::high_resolution_clock::now();
   SelfAdjointEigenSolver<MatrixXd> eigensolver(Hm);
   if (eigensolver.info() != Success) abort();
   auto eig = eigensolver.eigenvalues();
   auto vec = eigensolver.eigenvectors();
   cout << "eigenvalues:\n" << setprecision(10) << eig.head(3) << endl;
   //cout << "eigenvectors:\n" << scientific << vec << endl;
   auto time1 = chrono::high_resolution_clock::now();
   cout << " timing : " << setw(10) << fixed << setprecision(2) 
	<< chrono::duration_cast<chrono::milliseconds>(time1-time0).count()*0.001 << " s" << endl;

   // copy back 
   vector<double> v(vec.col(0).data(), vec.col(0).data()+dim);
   fock::coefficients(space2,v);

   vector<double> sigs(v.size());
   transform(v.cbegin(),v.cend(),sigs.begin(),
	     [](const double& x){return pow(x,2);});
   auto c2 = vec.col(0).array().pow(2);
   cout << sigs[0] << " " << c2[0] << endl;
  
   cout << vonNeumann_entropy(sigs) << endl;

   // sort?
   
   return 0;
}
