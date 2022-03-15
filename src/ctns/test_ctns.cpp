#include <iostream>
#include <iomanip>
#include <string>
#include "../io/input.h"
#include "../ci/ci_header.h"
#include "ctns_header.h"
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
   schd.read(fname);

   // we will use Tm to control Hnr/Hrel 
   //using Tm = double; // to do -> test more
   using Tm = complex<double>;
  
   // read integral
   integral::two_body<Tm> int2e;
   integral::one_body<Tm> int1e;
   double ecore;
   integral::load(int2e, int1e, ecore, schd.integral_file);
  
   // --- SCI ---
   int nroots = schd.sci.nroots;
   vector<double> es(nroots,0.0);
   onspace sci_space;
   vector<vector<Tm>> vs(nroots);
   
   if(!schd.sci.load){
      fci::sparse_hamiltonian<Tm> sparseH;
      sci::ci_solver(schd, sparseH, es, vs, sci_space, int2e, int1e, ecore);
      // pt2 for single root
      if(schd.sci.ifpt2){
         sci::pt2_solver(schd, es[0], vs[0], sci_space, int2e, int1e, ecore);
      }
      fci::ci_save(sci_space, es, vs);
   }else{
      fci::ci_load(sci_space, es, vs);
   }
   for(int i=0; i<nroots; i++){
      coeff_population(sci_space, vs[i]);
   }

   // --- Diagonalization in Fock space ---
   /*
   onspace fockSpace = get_fci_space(int1e.sorb);
   auto Hmat = fock::get_Hmat(fockSpace, int2e, int1e, ecore);
   vector<double> e(Hmat.rows());
   auto v(Hmat);
   eig_solver(Hmat, e, v); // Hc=ce
   for(int i=0; i<nroots; i++){
      std::cout << "e[" << i << "]=" << std::setprecision(10) << e[i] << std::endl;
      //coeff_population(fockSpace, v.col(i));
   } 
   // lih: parity=0 case
   // e[0]=-8.8890690229 
   // e[1]=-8.7645864467 
   // e[2]=-8.7645864425
   // e[3]=-8.7645864422
   */ 
 
   // --- CTNS --- 
   //ctns::comb<ctns::qkind::cZ2> icomb;
   ctns::comb<ctns::qkind::cN> icomb;
   //ctns::comb<ctns::qkind::cNK> icomb;
   
   // 1. dealing with topology 
   icomb.topo.read(schd.ctns.topology_file);
   icomb.topo.print();

   // 2. initialize right canonical form from SCI wavefunction
   fci::ci_truncate(sci_space, vs, schd.ctns.maxdets);

   if(!schd.ctns.rcanon_load){
      ctns::rcanon_init(icomb, sci_space, vs, schd.ctns.thresh_proj, schd.ctns.rdm_vs_svd);
      ctns::rcanon_save(icomb);
   }else{
      ctns::rcanon_load(icomb);
   }
   ctns::rcanon_check(icomb, schd.ctns.thresh_ortho);

   // 3. overlap
   const double thresh=1.e-6;
   // 3.1 <CTNS|CTNS>
   auto Sij_ci = fci::get_Smat(sci_space, vs);
   Sij_ci.print("Sij_ci");
   auto Sij_ctns = ctns::get_Smat(icomb);
   Sij_ctns.print("Sij");
   double diff_ctns = (Sij_ctns - Sij_ci).normF();
   cout << "\ncheck diff_Sij[ctns] = " << diff_ctns << endl;
   if(diff_ctns > thresh) tools::exit(string("error: diff_Sij[ctns] > thresh=")+to_string(thresh));
   // 3.2 <CI|CTNS>
   ctns::rcanon_CIcoeff_check(icomb, sci_space, vs);
   auto Sij_mix = ctns::rcanon_CIovlp(icomb, sci_space, vs);
   Sij_mix.print("Sij_mix");
   // check
   double diff_mix = (Sij_mix - Sij_ci).normF();
   cout << "\ncheck diff_Sij[mix] = " << diff_mix << endl;
   if(diff_mix > thresh) tools::exit(string("error: diff_Sij[mix] > thresh=")+to_string(thresh));

   // 4. compute Sd by sampling 
   int iroot = 0;
   double Sdiag0 = fock::coeff_entropy(vs[iroot]);
   double Sdiag1 = rcanon_Sdiag_exact(icomb,iroot);
   bool ifsample = false;
   if(ifsample){
      int nsample = 1.e5;
      double Sdiag2 = rcanon_Sdiag_sample(icomb,iroot,nsample);
      cout << "\niroot=" << iroot 
           << " Sdiag(exact)=" << Sdiag0
           << " Sdiag(brute-force)=" << Sdiag1 
           << " Sdiag(sample)=" << Sdiag2
           << endl;
   }

   schd.create_scratch(schd.scratch);

   // 5. Hij: construct renormalized operators
   const int alg_renorm = 0;
   auto Hij_ci = fci::get_Hmat(sci_space, vs, int2e, int1e, ecore);
   Hij_ci.print("Hij_ci",8);
   auto Hij_ctns = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch, alg_renorm);
   Hij_ctns.print("Hij_ctns",8);
   double diffH = (Hij_ctns - Hij_ci).normF();
   cout << "\ncheck diffH=" << diffH << endl;
   if(diffH > thresh) tools::exit(string("error: diffH > thresh=")+to_string(thresh));

   // 6. sweep optimization from current RCF 
   ctns::sweep_opt(icomb, int2e, int1e, ecore, schd, schd.scratch);

   // re-compute expectation value for optimized TNS
   auto Sij = ctns::get_Smat(icomb);
   Sij.print("Sij");
   auto Hij = ctns::get_Hmat(icomb, int2e, int1e, ecore, schd.scratch, alg_renorm);
   Hij.print("Hij",8);
   auto ovlp = rcanon_CIovlp(icomb, sci_space, vs);
   ovlp.print("ovlp");

   schd.remove_scratch(schd.scratch);

   return 0;
}
