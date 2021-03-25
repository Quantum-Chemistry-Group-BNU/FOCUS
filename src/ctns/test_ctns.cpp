#include <iostream>
#include <iomanip>
#include <string>
#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../io/input.h"
#include "../ci/fci.h"
#include "../ci/fci_rdm.h"
#include "../ci/sci.h"
#include "../ci/sci_pt2.h"
/*
#include "ctns_comb.h"
#include "ctns_comb_init.h"
#include "ctns_comb_alg.h"
#include "ctns_io.h"
#include "ctns_oper.h"
#include "ctns_opt.h"
*/
#include "tests_ctns.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_ctns(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_ctns" << endl;
   cout << tools::line_separator << endl;	

   // read input
   string fname = "input.dat";
   input::schedule schd;
   input::read(schd,fname);

/*
   // we will use DTYPE to control Hnr/Hrel 
   using DTYPE = complex<double>;
   //using DTYPE = double; // to do -> test more
  
   // read integral
   integral::two_body<DTYPE> int2e;
   integral::one_body<DTYPE> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
  
   // --- SCI ---
   int nroot = schd.nroots;
   vector<double> es(nroot,0.0);
   onspace sci_space;
   vector<vector<DTYPE>> vs(nroot);
   
   if(!schd.ciload){
      fci::sparse_hamiltonian<DTYPE> sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      // pt2 for single root
      if(schd.ifpt2){
         sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, vs);
   }else{
      fci::ci_load(sci_space, vs);
   }
   for(int i=0; i<nroot; i++){
      coeff_population(sci_space, vs[i]);
   }
   // truncate CI coefficients
   const bool ifortho = false; //true; 
   fci::ci_truncate(sci_space, vs, schd.maxdets, ifortho);

   // --- Comb TNS (CTNS) ---   
   // 1. dealing with topology 
   ctns::topology topo(schd.topology_file);
   ctns::comb<DTYPE> icomb(topo);
   icomb.topo.print();
   
   schd.create_scratch();

   if(!schd.combload){
      // 2. initialize right canonical form from SCI wavefunction 
      ctns::rcanon_init(icomb, sci_space, vs, schd.thresh_proj);
      ctns::rcanon_check(icomb, schd.thresh_ortho, ifortho);
      
      const double thresh=1.e-6;

      // 3. algorithm: check overlap with CI 
      auto ovlp = ctns::rcanon_CIovlp(icomb, sci_space, vs);
      ovlp.print("CIovlp");
      // check overlap
      auto Smat = fci::get_Smat(sci_space, vs);
      Smat.print("Smat");
      auto Sij = ctns::get_Smat(icomb);
      Sij.print("Sij");
      double diff = normF(Smat-Sij);
      cout << "diff_Sij=" << diff << endl;
      if(diff > thresh){
         cout << "error: diff_Sij > thresh=" << thresh << endl;
         exit(1);
      }

      // 4. check Hij 
      auto Hmat = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
      Hmat.print("Hmat",8);
      auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
      Hij.print("Hij",8);
      diff = normF(Hmat-Hij);
      cout << "diff_Hij=" << diff << endl;
      if(diff > thresh){ 
         cout << "error: diff_Hij > thresh=" << thresh << endl;
         exit(1);
      }
  
      // 5. optimization from current RCF 
      ctns::opt_sweep(icomb, int2e, int1e, ecore, schd);
      exit(1);

      ctns::rcanon_save(icomb);
   }else{
      ctns::rcanon_load(icomb);
   }

   // re-compute expectation value for optimized TNS
   auto Sij = ctns::get_Smat(icomb);
   Sij.print("Sij");
   auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
   Hij.print("Hij",8);
   auto ovlp = rcanon_CIovlp(icomb, sci_space, vs);
   ovlp.print("ovlp");

   // 6. compute Sd by sampling 
   int istate = 0, nsample = 1.e5;
   double Sdiag1 = rcanon_Sdiag_sample(icomb,istate,nsample);
   double Sdiag2 = rcanon_Sdiag_exact(icomb,istate);
   cout << "istate=" << istate 
        << " Sdiag(sample)=" << Sdiag1 
        << " Sdiag(exact)=" << Sdiag2 
        << endl;

   schd.remove_scratch();
*/

   return 0;
}
