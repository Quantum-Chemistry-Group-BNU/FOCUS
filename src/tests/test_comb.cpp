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
#include "../utils/tns_comb.h"
#include "../utils/tns_oper.h"
#include "../settings/global.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_comb(){
   cout << endl;	
   cout << global::line_separator << endl;	
   cout << "tests::test_comb" << endl;
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
   }else{
      fci::ci_load(sci_space, vs);
   }
   // check 
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
   }
   // truncate CI coefficients
   sci::ci_truncate(sci_space, vs, schd.maxdets);
 
   // comb tensor networks
   tns::comb comb;
   comb.read_topology(schd.topology_file);
   comb.init();
   comb.print();

   if(!schd.combload){
      comb.rcanon_init(sci_space, vs, schd.thresh_proj, schd.thresh_ortho);
      comb.rcanon_save();
   }else{
      comb.rcanon_load();
   }
   
   // check overlap
   auto ovlp = comb.rcanon_ovlp(sci_space, vs);
   ovlp.print("ovlp");

   // check energy
   auto Smat = fci::get_Smat(sci_space, vs);
   auto Hmat = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
   Smat.print("Smat");
   Hmat.print("Hmat");
  
   auto Sij = tns::get_Sij(comb, comb);
   auto Hij = tns::get_Hij(comb, comb, int2e, int1e, ecore);
   Sij.print("Sij");
   Hij.print("Hij");

   // optimization

   return 0;
}
