#include "../core/onspace.h"
#include "../core/integral.h"
#include "../core/analysis.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../ci/fci.h"
#include "../ci/fci_rdm.h"
#include "../ci/sci.h"
#include "../ci/sci_pt2.h"
#include "../ctns/ctns_topo.h"
#include "../io/input.h"
#include <iostream>
#include <iomanip>
#include <string>
#include "tests.h"

using namespace std;
using namespace fock;
using namespace linalg;

int tests::test_ctns(){
   cout << endl;	
   cout << tools::line_separator << endl;	
   cout << "tests::test_ctns" << endl;
   cout << tools::line_separator << endl;	

   // read input
   string fname = "ctns.dat";
   input::schedule schd;
   input::read(schd,fname);

   // we will use DTYPE to control Hnr/Hrel 
   using DTYPE = double;
   
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
   const bool ifortho = true;
   fci::ci_truncate(sci_space, vs, schd.maxdets, ifortho);

   // --- Comb TNS (CTNS) ---
/*
   // --- 1. dealing with topology ---
   ctns::topology topo;
   topo.read(schd.topology_file);
   topo.print();

   schd.create_scratch();

   if(!schd.combload){

      // --- 2. initialize right canonical form from SCI wavefunction --- 
      comb.rcanon_init(sci_space, vs, schd.thresh_proj);
      comb.rcanon_check(schd.thresh_ortho, ifortho);
      
      const double thresh=1.e-6;
      // --- 3. algorithm: check overlap with CI --- 
      auto ovlp = comb.rcanon_CIovlp(sci_space, vs);
      ovlp.print("ovlp");
      // check self-overlap
      auto Smat = fci::get_Smat(sci_space, vs);
      Smat.print("Smat");
      auto Sij = tns::get_Smat(comb, comb);
      Sij.print("Sij");
      double diff = normF(Smat-Sij);
      cout << "diff_Sij=" << diff << endl;
      if(diff > thresh){
         cout << "error: diff_Sij > thresh=" << thresh << endl;
         exit(1);
      }
  
      //// check rdm1 & Bpq
      //int k = int1e.sorb;
      //linalg::matrix rdm1(k,k); 
      //fci::get_rdm1(sci_space, vs[0], vs[0], rdm1);
      //rdm1.save("fci_rdm1a");
      //fci::get_rdm1(sci_space, vs[2], vs[0], rdm1);
      //rdm1.save("fci_rdm1b");
      //fci::get_rdm1(sci_space, vs[1], vs[2], rdm1);
      //rdm1.save("fci_rdm1c");

      // --- 4. algorithm: check Hij ---
      auto Hmat = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
      Hmat.print("Hmat",8);
      Hmat.save("fci_Hmat");
      auto Hij = tns::get_Hmat(comb, comb, int2e, int1e, ecore, schd.scratch);
      Hij.print("Hij",8);
      diff = normF(Hmat-Hij);
      cout << "diff_Hij=" << diff << endl;
      if(diff > thresh){ 
         cout << "error: diff_Hij > thresh=" << thresh << endl;
         exit(1);
      }
  
      // --- 5. optimization from current RCF ---  
      tns::opt_sweep(schd, comb, int2e, int1e, ecore);
      comb.rcanon_save();

   }else{
      comb.rcanon_load();
   }

   // re-compute expectation value for optimized TNS
   auto Sij = tns::get_Smat(comb, comb);
   Sij.print("Sij");
   auto Hij = tns::get_Hmat(comb, comb, int2e, int1e, ecore, schd.scratch);
   Hij.print("Hij",8);
 
   schd.remove_scratch();

   // check with SCI
   auto ovlp = comb.rcanon_CIovlp(sci_space, vs);
   ovlp.print("ovlp");
   
   // --- 6. compute Sd by sampling ---
   int nsample = 1.e5, istate = 0, nprt = 10;
   double Sd = comb.rcanon_sampling_Sd(nsample,istate,nprt);
   cout << "istate=" << istate << " Sd(estimate)=" << Sd << endl;
   // only for small system - exact computation
   comb.rcanon_sampling_check(istate);
*/

   return 0;
}
