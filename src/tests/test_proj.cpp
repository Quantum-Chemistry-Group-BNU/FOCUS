#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../utils/fci.h"
#include "../utils/fci_rdm.h"
#include "../utils/sci.h"
#include "../settings/global.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_proj(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_proj" << endl;
   cout << global::line_separator << endl;	

   // read input
   string fname = "input.dat";
   input::schedule schd;
   input::read_input(schd,fname);

   // read integral
   integral::two_body int2e;
   integral::one_body int1e;
   double ecore;
   integral::read_fcidump(int2e, int1e, ecore, 
		   	  schd.integral_file,
		    	  schd.integral_type);
  
   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);

   // selected CI
   onspace sci_space;
   vector<vector<double>> vs(nroot);
   
   if(!schd.ciload){
      fci::sparse_hamiltonian sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      sparseH.analysis();
      // pt2 for single root
      if(schd.ifpt2){
         sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, vs);
      cout << vs[0][0] << endl;
   }else{
      fci::ci_load(sci_space, vs);
      int dim = vs[0].size();
      cout << "dim=" << dim << endl;
      cout << vs[0][0] << " " << vs[0][dim-1] << endl;
      cout << vs[nroot-1][0] << " " << vs[nroot-1][dim-1] << endl;
   }
      
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
      exit(1);
   }

   // compute rdm1
   int k = int1e.sorb; 
   linalg::matrix rdm1(k,k);
   for(int i=0; i<nroot; i++){
      linalg::matrix rdm1t(k,k);
      fci::get_rdm1(sci_space,vs[i],vs[i],rdm1t);
      rdm1 += rdm1t*(1.0/nroot);
   }
   // natural orbitals
   linalg::matrix u;
   vector<double> occ;
   fci::get_natorb_nr(rdm1,u,occ);
   exit(1);

   return 0;
}
