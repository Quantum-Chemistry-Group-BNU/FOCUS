#include <iostream>
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../settings/global.h"
#include "../utils/fci.h"
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "tests.h"

using namespace std;
using namespace fock;

void compare_eigs(vector<double>& es, 
		  vector<double>& es1){
   int nroot = es.size();
   cout << "\nCheck difference:" << endl;
   cout << defaultfloat << setprecision(10);
   for(int i=0; i<nroot; i++){
      cout << "i=" << i 
	   << " e=" << es[i] << " " << es1[i] 
	   << " diff=" << es1[i]-es[i] << endl;
      assert(abs(es[i]-es1[i])<1.e-10);
   }
}

int tests::test_fci(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_fci" << endl;
   cout << global::line_separator << endl;	
   
   double thresh = 1.e-5;
  
   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore0;

   int k, ne;
   integral::read_fcidump(int2e, int1e, ecore0, "../fcidump/FCIDUMP_lih");
   k = 6*2; ne = 4; // lih
   onspace fci_space = get_fci_space(k/2,ne/2,ne/2);
   int dim = fci_space.size();

   double ecore = 0.0;
   integral::two_body int2e_tmp;
   integral::one_body int1e_tmp;
   int nroot = 1; 
   vector<double> es(nroot,0.0);
   linalg::matrix vs(dim,nroot);
   vector<double> es1(nroot,0.0);
   linalg::matrix vs1(dim,nroot);

   //----------------------------------------------
   // 1. integrals: AA,AAAA 
   //----------------------------------------------
   int1e_tmp = int1e.get_AA();
   int2e_tmp = int2e.get_AAAA();
   //----------------------------------------------
   ci_solver(es, vs, fci_space, int2e_tmp, int1e_tmp, ecore);
   fci::ci_solver(es1, vs1, fci_space, int2e_tmp, int1e_tmp, ecore);
   compare_eigs(es, es1);
   double e1e0 = -6.043786524747;
   assert(abs(es[0]-e1e0) < thresh);

   //----------------------------------------------
   // 2. integrals: AA,AAAA + BB,BBBB 
   //----------------------------------------------
   int1e_tmp = int1e.get_AA() + int1e.get_BB();
   int2e_tmp = int2e.get_AAAA() + int2e.get_BBBB();
   //----------------------------------------------
   ci_solver(es, vs, fci_space, int2e_tmp, int1e_tmp, ecore);
   fci::ci_solver(es1, vs1, fci_space, int2e_tmp, int1e_tmp, ecore);
   compare_eigs(es, es1);
   assert(abs(es[0]-2*e1e0) < thresh);

/*
   //----------------------------------------------
   // 3. integrals: BBAA 
   //----------------------------------------------
   int1e_tmp.clear();
   int2e_tmp = int2e.get_BBAA();
   //----------------------------------------------
   ci_solver(es, vs, fci_space, int2e_tmp, int1e_tmp, ecore);
   fci::ci_solver(es1, vs1, fci_space, int2e_tmp, int1e_tmp, ecore);
   compare_eigs(es, es1);
   assert(abs(es[0]-0.9667157752) < thresh);
*/

   //----------------------------------------------
   // 4. integrals: AA,AAAA + BB,BBBB + BBAA 
   //----------------------------------------------
   int1e_tmp = int1e.get_AA() + int1e.get_BB();
   int2e_tmp = int2e.get_AAAA() + int2e.get_BBBB() + int2e.get_BBAA();
   //----------------------------------------------
   ecore = ecore0; 
   ci_solver(es, vs, fci_space, int2e_tmp, int1e_tmp, ecore);
   fci::ci_solver(es1, vs1, fci_space, int2e_tmp, int1e_tmp, ecore);
   compare_eigs(es, es1);
   double e0 = -7.873881390340;
   assert(abs(es[0]-e0) < thresh);

   //----------------------------------------------
   // check eigenvectors
   //----------------------------------------------
   cout << "|vs1-vs|=" << normF(vs1-vs) << endl;
   /*
   for(int i=0; i<dim; i++){
      if(abs(vs(i,0)) < thresh && abs(vs1(i,0)) < thresh) continue;
      cout << "i=" << i 
	   << " v=" << vs(i,0) << " " << vs1(i,0) 
	   << " diff=" << vs1(i,0)-vs(i,0) << endl;
   }
   */
   // analysis 
   vector<double> v0i(vs.col(0),vs.col(0)+dim);
   fock::coefficients(fci_space, v0i);
   vector<double> sigs(v0i.size());
   transform(v0i.cbegin(),v0i.cend(),sigs.begin(),
	     [](const double& x){return pow(x,2);}); // pi=|ci|^2
   auto SvN = vonNeumann_entropy(sigs);
   cout << "p0=" << sigs[0] << endl;
   cout << "SvN=" << SvN  << endl;
   assert(abs(sigs[0]-0.9805968962) < thresh);
   assert(abs(SvN-0.1834419989) < thresh);

   return 0;
}
