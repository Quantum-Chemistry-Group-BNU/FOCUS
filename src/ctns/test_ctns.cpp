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
#include "ctns_comb.h"
#include "ctns_init.h"
#include "ctns_io.h"
#include "ctns_ovlp.h"
#include "ctns_oper.h"
#include "ctns_sweep.h"
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

   // we will use DTYPE to control Hnr/Hrel 
   using DTYPE = double; // to do -> test more
   //using DTYPE = complex<double>;
  
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
   const bool ifortho = false; 
   fci::ci_truncate(sci_space, vs, schd.maxdets, ifortho);

   // --- CTNS --- 
   
   // 1. dealing with topology 
   ctns::topology topo(schd.topology_file);
   topo.print();

   // 2. initialize right canonical form from SCI wavefunction
   ctns::comb<ctns::kind::rNSz> icomb(topo);
   //ctns::comb<ctns::kind::cN> icomb(topo);
   //ctns::comb<ctns::kind::cNK> icomb(topo);

   if(!schd.combload){
      ctns::rcanon_init(icomb, sci_space, vs, schd.thresh_proj);
      ctns::rcanon_save(icomb);
   }else{
      ctns::rcanon_load(icomb);
   }
   ctns::rcanon_check(icomb, schd.thresh_ortho, ifortho);

   // 3. overlap
   const double thresh=1.e-6;
   // <CI|CI>
   auto Sij_ci = fci::get_Smat(sci_space, vs);
   Sij_ci.print("Sij_ci");
   // <CTNS|CTNS>
   auto Sij_ctns = ctns::get_Smat(icomb);
   Sij_ctns.print("Sij");
   // check
   double diff_ctns = normF(Sij_ctns - Sij_ci);
   cout << "\ncheck diff_Sij[ctns] = " << diff_ctns << endl;
   if(diff_ctns > thresh){
      cout << "error: diff_Sij[ctns] > thresh=" << thresh << endl;
      exit(1);
   }
   // <CI|CTNS>
   ctns::rcanon_CIcoeff_check(icomb, sci_space, vs);
   auto Sij_mix = ctns::rcanon_CIovlp(icomb, sci_space, vs);
   Sij_mix.print("Sij_mix");
   // check
   double diff_mix = normF(Sij_mix - Sij_ci);
   cout << "\ncheck diff_Sij[mix] = " << diff_mix << endl;
   if(diff_mix > thresh){
      cout << "error: diff_Sij[mix] > thresh=" << thresh << endl;
      exit(1);
   }

   // 4. compute Sd by sampling 
   int istate = 0, nsample = 1.e5;
   double Sdiag0 = fock::coeff_entropy(vs[istate]);
   double Sdiag1 = rcanon_Sdiag_exact(icomb,istate);
   bool ifsample = false;
   if(ifsample){
      double Sdiag2 = rcanon_Sdiag_sample(icomb,istate,nsample);
      cout << "\nistate=" << istate 
           << " Sdiag(exact)=" << Sdiag0
           << " Sdiag(brute-force)=" << Sdiag1 
           << " Sdiag(sample)=" << Sdiag2
           << endl;
   }

   schd.create_scratch();
   
   // 5. Hij
   auto Hij_ci = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
   Hij_ci.print("Hij_ci",8);
   auto Hij_ctns = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
   Hij_ctns.print("Hij_ctns",8);
   double diffH = normF(Hij_ctns - Hij_ci);
   cout << "\ncheck diffH=" << diffH << endl;
   if(diffH > thresh){ 
      cout << "error: diffH > thresh=" << thresh << endl;
      exit(1);
   }
 
   // 6. optimization from current RCF 
   ctns::sweep_opt(icomb, int2e, int1e, ecore, schd);

   // re-compute expectation value for optimized TNS
   auto Sij = ctns::get_Smat(icomb);
   Sij.print("Sij");
   auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
   Hij.print("Hij",8);
   auto ovlp = rcanon_CIovlp(icomb, sci_space, vs);
   ovlp.print("ovlp");

   schd.remove_scratch();

   return 0;
}
