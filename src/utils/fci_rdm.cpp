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

// c1[i]<Di|p0^+p0|Di>c2[i]
void fci::get_rdm1_diag(const onspace& space,
	                const vector<double>& civec1,
		        const vector<double>& civec2,
		        matrix& rdm1){
   for(size_t i=0; i<space.size(); i++){
      // c1[i]<Di|p^+q|Di>c2[i]
      vector<int> olst;
      space[i].get_olst(olst);
      for(int p : olst){
         rdm1(p,p) += civec1[i]*civec2[i];
      }
   }
}

// c1[i]<Di|p0^+p1^+p1p0|Di>c2[i]
void fci::get_rdm2_diag(const onspace& space,
	                const vector<double>& civec1,
		        const vector<double>& civec2,
		        matrix& rdm2){
   for(size_t i=0; i<space.size(); i++){
      vector<int> olst;
      space[i].get_olst(olst);
      for(int idx=0; idx<olst.size(); idx++){
         auto p0 = olst[idx]; 
	 for(int jdx=0; jdx<idx; jdx++){
            auto p1 = olst[jdx];
	    auto p01 = tools::canonical_pair0(p0,p1);
	    rdm2(p01,p01) += civec1[i]*civec2[i]; 
	 }
      }
   }
}

// <Psi1|p^+q|Psi2> (NR case)
void fci::get_rdm1(const onspace& space,
 		   const vector<double>& civec1,
		   const vector<double>& civec2,
		   matrix& rdm1){
   cout << "\nfci:get_rdm1" << endl;
   bool debug_rdm1 = false;
   auto t0 = global::get_time();
   // setup product_space
   product_space pspace(space);
   // setupt coupling_table
   coupling_table ctabA(pspace.umapA);
   coupling_table ctabB(pspace.umapB);
   // diagonal term
   get_rdm1_diag(space, civec1, civec2, rdm1);
   // off-diagonal term:
   // <I_A,I_B|pA^+qA|J_A,J_B> - essentially follow from construction of H_connect
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
	 int p[1], q[1];
         pspace.spaceA[ia].diff_orb(pspace.spaceA[ja],p,q);
	 int p0 = 2*p[0], q0 = 2*q[0];
         for(const auto& pb : pspace.rowA[ia]){
            int ib = pb.first;
            int i = pb.second;
            int j = pspace.dpt[ja][ib];
            if(j>i){
               auto sgn = space[i].parity(p0)*space[j].parity(q0);
               rdm1(p0,q0) += sgn*civec1[i]*civec2[j];
               rdm1(q0,p0) += sgn*civec1[j]*civec2[i];
            }
	 }
      }
   }
   // <I_A,I_B|pB^+qB|J_A,J_B>
   for(int ib=0; ib<pspace.dimB; ib++){
      for(int jb : ctabB.C11[ib]){
         int p[1], q[1];
	 pspace.spaceB[ib].diff_orb(pspace.spaceB[jb],p,q);
         int p0 = 2*p[0]+1, q0 = 2*q[0]+1;
         for(const auto& pa : pspace.colB[ib]){
            int ia = pa.first;
            int i = pa.second;
            int j = pspace.dpt[ia][jb];
            if(j>i){
               double sgn = space[i].parity(p0)*space[j].parity(q0); 
               rdm1(p0,q0) += sgn*civec1[i]*civec2[j];
               rdm1(q0,p0) += sgn*civec1[j]*civec2[i];
            }
	 }
      }
   } // ib
   // compute trace
   double tr = 0.0;
   for(int k=0; k<rdm1.rows(); k++){
      tr += rdm1(k,k);
   }
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

// contribution from two states differ by (1,1): <p^+k^+kq>
void fci::get_rdm2S(const onstate& stateI,
		    const onstate& stateJ,
		    const double& civec1I,
		    const double& civec1J,
		    const double& civec2I,
		    const double& civec2J,
		    matrix& rdm2){
   int p[1], q[1];
   stateI.diff_orb(stateJ,p,q);
   auto p0 = p[0];
   auto q0 = q[0];
   auto sgn0 = stateI.parity(p0)*stateJ.parity(q0);
   vector<int> olst;
   stateI.get_olst(olst);
   for(int idx=0; idx<olst.size(); idx++){
      auto p1 = olst[idx];
      if(p1 == p0) continue; 
      auto sgn = sgn0;
      auto p01 = tools::canonical_pair0(p0,p1);
      if(p0 < p1) sgn *= -1; // sign coming from ordering of operators
      // p1 must be not identical to q0, otherwise it cannot be in olst 
      auto q01 = tools::canonical_pair0(q0,p1);
      if(q0 < p1) sgn *= -1; 
      rdm2(p01,q01) += sgn*civec1I*civec2J;
      rdm2(q01,p01) += sgn*civec1J*civec2I;
   } // idx
}

// contribution from two states differ by (2,2): <p^+p^+qq>
void fci::get_rdm2D(const onstate& stateI,
		    const onstate& stateJ,
		    const double& civec1I,
		    const double& civec1J,
		    const double& civec2I,
		    const double& civec2J,
		    matrix& rdm2){
   int p[2], q[2];
   stateI.diff_orb(stateJ,p,q);
   auto p01 = tools::canonical_pair0(p[0],p[1]);
   auto q01 = tools::canonical_pair0(q[0],q[1]);
   auto sgn = stateI.parity(p[0])*stateI.parity(p[1])
            * stateJ.parity(q[0])*stateJ.parity(q[1]);
   rdm2(p01,q01) += sgn*civec1I*civec2J;
   rdm2(q01,p01) += sgn*civec1J*civec2I;
}

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1)
void fci::get_rdm2(const onspace& space,
 		   const vector<double>& civec1,
		   const vector<double>& civec2,
		   matrix& rdm2){
   cout << "\nfci:get_rdm2" << endl;
   bool debug_rdm2 = false;
   auto t0 = global::get_time();
   // setup product_space
   product_space pspace(space);
   // setupt coupling_table
   coupling_table ctabA(pspace.umapA);
   coupling_table ctabB(pspace.umapB);
   // diagonal term
   get_rdm2_diag(space, civec1, civec2, rdm2);
   // 1. (C11+C22)_A*C00_B:
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
	       get_rdm2S(space[i],space[j],civec1[i],civec1[j],
			 civec2[i],civec2[j],rdm2);
	    }
	 }
      }
      // AAAA = pA^+pA'^+qAqA'
      for(int ja : ctabA.C22[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
	       get_rdm2D(space[i],space[j],civec1[i],civec1[j],
			 civec2[i],civec2[j],rdm2);
	    }
	 }
      }
   } // ia
   // 2. C00_A*(C11+C22)_B:
   for(int ib=0; ib<pspace.dimB; ib++){
      for(const auto& pa : pspace.colB[ib]){
         int ia = pa.first;
         int i = pa.second;
	 // BSSB: S={A,B}
         for(int jb : ctabB.C11[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       get_rdm2S(space[i],space[j],civec1[i],civec1[j],
			 civec2[i],civec2[j],rdm2);
	    }
	 }
	 // BBBB
         for(int jb : ctabB.C22[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       get_rdm2D(space[i],space[j],civec1[i],civec1[j],
			 civec2[i],civec2[j],rdm2);
	    }
	 }
      }
   } // ib
   // 3. C11_A*C11_B: 
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
         for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
            int i = pb.second;
   	    for(int jb : ctabB.C11[ib]){
   	       int j = pspace.dpt[ja][jb];
   	       if(j>i){
	          get_rdm2D(space[i],space[j],civec1[i],civec1[j],
	           	    civec2[i],civec2[j],rdm2);
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
   // compute trace
   double tr = 0.0;
   for(int k=0; k<rdm2.rows(); k++){
      tr += rdm2(k,k);
   }
   tr = 2*tr;
   cout << "tr(rdm2)=" << tr << " normalized to N(N-1)" << endl;
   auto t1 = global::get_time();
   cout << "timing for rdm2 : " << setprecision(2) 
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

// <Psi|p0^+p1^+q1q0|Psi> (p0>p1, q0>q1) using sparseH
// which contains the computed connection information  
void fci::make_rdm2(const onspace& space,
	 	    const sparse_hamiltonian& sparseH,
	            const vector<double>& civec1,
		    const vector<double>& civec2,
		    matrix& rdm2){
   cout << "\nfci:make_rdm2" << endl;
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
      for(const auto& pj : sparseH.connect[i]){
         int j = get<0>(pj);
	 long ph = get<2>(pj);
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
}
