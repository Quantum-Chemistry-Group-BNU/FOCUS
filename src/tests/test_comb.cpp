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
#include "../utils/tns_alg.h"
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
   comb.topo_read(schd.topology_file);
   comb.topo_init();
   comb.topo_print();

   if(!schd.combload){
      comb.rcanon_init(sci_space, vs, schd.thresh_proj, schd.thresh_ortho);
      comb.rcanon_save();
   }else{
      comb.rcanon_load();
   }
   
   // check overlap with CI
   auto ovlp = comb.rcanon_CIovlp(sci_space, vs);
   ovlp.print("ovlp");
   
   // check self-overlap
   auto Smat = fci::get_Smat(sci_space, vs);
   Smat.print("Smat");
   auto Sij = tns::get_Sij(comb, comb);
   Sij.print("Sij");

   // check rdm1
   int k = int1e.sorb;
   linalg::matrix rdm1(k,k); 
   fci::get_rdm1(sci_space, vs[0], vs[0], rdm1);
   rdm1.save("fci_rdm1a");
   fci::get_rdm1(sci_space, vs[2], vs[0], rdm1);
   rdm1.save("fci_rdm1b");
   fci::get_rdm1(sci_space, vs[1], vs[2], rdm1);
   rdm1.save("fci_rdm1c");
  
   schd.create_scratch();
   int1e.set_zeros();
   //int2e.set_zeros();

   auto Hmat = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
   cout << "ecore=" << ecore << endl;
   Hmat.print("Hmat");
   Hmat.save("fci_Hmat");

   tns::oper_env_right(comb, comb, int2e, int1e, schd.scratch);
   
   // check energy
   //auto Hij = tns::get_Hij(comb, comb, int2e, int1e, ecore);
   //Hij.print("Hij");
   //schd.remove_scratch();

   return 0;
}
