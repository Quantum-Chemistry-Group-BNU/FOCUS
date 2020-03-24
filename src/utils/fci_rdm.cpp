#include "../settings/global.h"
#include "../core/hamiltonian.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/analysis.h" // rdm debug
#include "fci.h"
#include "fci_rdm.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;

// <Psi1|p^+q|Psi2> (NR case)
void fci::get_rdm1(const onspace& space,
 		   const vector<double>& civec1,
		   const vector<double>& civec2,
		   matrix& rdm1){
   cout << "\nfci:get_rdm1" << endl;
   bool debug_rdm1 = false;
   auto t0 = global::get_time();
  
   // setup product_space
   product_space pspace;
   pspace.get_pspace(space);
   // setupt coupling_table
   coupling_table ctabA, ctabB;
   ctabA.get_C11(pspace.spaceA);
   ctabB.get_C11(pspace.spaceB);
   // diagonal term: c1[i]<Di|p^+p|Di>c2[i] (i=j)
   for(size_t i=0; i<space.size(); i++){
      vector<int> olst;
      space[i].get_olst(olst);
      for(int p : olst){
         rdm1(p,p) += civec1[i]*civec2[i];
      }
   }
   // off-diagonal term:
   for(int ia=0; ia<pspace.dimA; ia++){
      for(const auto& pib : pspace.rowA[ia]){
         int ib = pib.first;
         int i = pib.second;
         // 1. <I_A,I_B|pA^+qA|J_A,J_B> = <I_A|pA^+qA|J_A><I_B|J_B> 
         //    essentially follow from construction of H_connect
     	 for(const auto& pja : pspace.colB[ib]){ 
	    int ja = pja.first;
	    int j = pja.second;
	    if(j <= i) continue;
	    auto search = ctabA.C11[ia].find(ja);
	    if(search != ctabA.C11[ia].end()){
	       int p[1], q[1];
               space[i].diff_orb(space[j],p,q);
               auto sgn = space[i].parity(p[0])*space[j].parity(q[0]);
               rdm1(p[0],q[0]) += sgn*civec1[i]*civec2[j];
               rdm1(q[0],p[0]) += sgn*civec1[j]*civec2[i];
            }
	 } // ja
	 // 2. <I_A,I_B|pB^+qB|J_A,J_B> = <I_A|J_A><I_B|pB^+qB|J_B>
         for(const auto& pjb : pspace.rowA[ia]){
	    int jb = pjb.first;
	    int j = pjb.second;
	    if(j <= i) continue; 
	    auto search = ctabB.C11[ib].find(jb);
	    if(search != ctabB.C11[ib].end()){
	       int p[1], q[1];
               space[i].diff_orb(space[j],p,q);
               auto sgn = space[i].parity(p[0])*space[j].parity(q[0]);
               rdm1(p[0],q[0]) += sgn*civec1[i]*civec2[j];
               rdm1(q[0],p[0]) += sgn*civec1[j]*civec2[i];
	    }
	 } // jb
      } // ib
   } // ia
   double tr = rdm1.trace();
   cout << "tr(rdm1)=" << tr << " normalized to N" << endl;
   auto t1 = global::get_time();
   cout << "timing for fci:get_rdm1 : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   // debug by comparing against the brute-force implementation 
   if(debug_rdm1){
      matrix rdm1b(rdm1.rows(),rdm1.cols());
      fock::get_rdm1(space, civec1, civec2, rdm1b);
      auto t2 = global::get_time();
      cout << "timing for rdm1 : " << setprecision(2) 
   	   << global::get_duration(t2-t1) << " s" << endl;
      auto rdm1_diff = normF(rdm1b - rdm1);
      cout << "rdm1_diff=" << rdm1_diff << endl;
      if(rdm1_diff>1.e-8) exit(1);
   }
}

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1) using sparseH
// which contains the computed connection information  
void fci::get_rdm2(const sparse_hamiltonian& sparseH,
		   const onspace& space,
	           const vector<double>& civec1,
		   const vector<double>& civec2,
		   matrix& rdm2){
   cout << "\nfci:get_rdm2" << endl;
   bool debug_rdm2 = false;
   auto t0 = global::get_time();
   
   int k = space[0].size();
   for(int i=0; i<sparseH.dim; i++){
      // diagonal term
      vector<int> olst;
      space[i].get_olst(olst);
      for(int idx=0; idx<olst.size(); idx++){
         int p0 = olst[idx]; 
	 for(int jdx=0; jdx<idx; jdx++){
            int p1 = olst[jdx];
	    int p01 = tools::canonical_pair0(p0,p1);
	    rdm2(p01,p01) += civec1[i]*civec2[i]; 
	 }
      }
      // off-diagonal term: ci*<Di|p0^+p1^+q1q0|Dj>cj (j != i)
      for(int jdx=0; jdx<sparseH.connect[i].size(); jdx++){
	 int j = sparseH.connect[i][jdx];
	 long ph = sparseH.diff[i][jdx];
    	 int sgn0 = ph>0? 1 : -1;
    	 ph = std::abs(ph);
    	 int p0 = ph%k;
    	 int q0 = (ph/k)%k;
	 // single excitations
	 if(ph/k/k == 0){
	    for(const int& p1 : olst){
               if(p1 == p0) continue;
               int p01 = tools::canonical_pair0(p0,p1);
               int q01 = tools::canonical_pair0(q0,p1);
	       int sgn = ((p0<p1)^(q0<p1))? -sgn0 : sgn0; 
               rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
               rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
            }
	 // double excitations   
	 }else{
	    int p1 = (ph/k/k)%k;
	    int q1 = ph/k/k/k;
	    int p01 = p0*(p0-1)/2+p1;
	    int q01 = q0*(q0-1)/2+q1;
            rdm2(p01,q01) += sgn0*civec1[i]*civec2[j];
            rdm2(q01,p01) += sgn0*civec1[j]*civec2[i];
	 }
      }	// Dj     
   } // Di
   double tr = 2.0*rdm2.trace();
   cout << "tr(rdm2)=" << tr << " normalized to N(N-1)" << endl;
   auto t1 = global::get_time();
   cout << "timing for fci:get_rdm2 : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
   // debug by comparing against the brute-force implementation 
   if(debug_rdm2){
      matrix rdm2b(rdm2.rows(),rdm2.cols());
      fock::get_rdm2(space, civec1, civec2, rdm2b);
      auto t2 = global::get_time();
      cout << "timing for rdm2 : " << setprecision(2) 
   	   << global::get_duration(t2-t1) << " s" << endl;
      auto rdm2_diff = normF(rdm2b - rdm2);
      cout << "rdm2_diff=" << rdm2_diff << endl;
      if(rdm2_diff>1.e-8) exit(1);
   }
}

// natural orbital
void fci::get_natural_nr(const matrix& rdm1,
		         matrix& u,
		         vector<double>& occ){
   int k1 = rdm1.rows()/2;
   matrix u1(k1,k1);
   occ.resize(k1);
   for(int j=0; j<k1; j++){
      for(int i=0; i<k1; i++){
         u1(i,j) = rdm1(2*i,2*j) + rdm1(2*i+1,2*j+1);
      }
   }
   u = -1.0*u1;
   // diagonalize spin-averaged dm
   eigen_solver(u, occ);
   transform(occ.begin(), occ.end(), occ.begin(),
             [](const double x){ return -x; });
   double ne = 0.0;
   cout << "\nfci::get_natural_nr k/2=" << rdm1.rows()/2 << endl;
   for(int i=0; i<k1; i++){
      cout << setw(3) << i 
	   << " :" << fixed << setw(7) << setprecision(4) << occ[i] 
	   << endl;
      ne += occ[i];
   }
   cout << "no. of electrons=" << ne << endl;
}
