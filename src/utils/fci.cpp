#include <functional>
#include <algorithm>
#include "../settings/global.h"
#include "../core/dvdson.h"
#include "../core/hamiltonian.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/analysis.h" // rdm debug
#include "fci.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;

// constructor
product_space::product_space(const onspace& space){
   int dim = space.size();
   int udxA = 0, udxB = 0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   for(int i=0; i<dim; i++){
      // even 
      auto strA = space[i].get_even();
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
         auto pr = umapA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 rowA.resize(udxA); // make space for new row
      };
      // odd 
      auto strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 colB.resize(udxB);
      }
      rowA[itA->second].emplace_back(itB->second,i);
      colB[itB->second].emplace_back(itA->second,i);
   }
   // nnzA
   dimA = udxA;
   nnzA.resize(dimA);
   for(int ia=0; ia<dimA; ia++) nnzA[ia] = rowA[ia].size();
   // nnzB
   dimB = udxB;
   nnzB.resize(dimB);
   for(int ib=0; ib<dimB; ib++) nnzB[ib] = colB[ib].size();	  
   // dpt
   dpt.resize(dimA); 
   for(int ia=0; ia<dimA; ia++){
      dpt[ia].resize(dimB,-1);
      for(int ib=0; ib<nnzA[ia]; ib++){
	 int b = rowA[ia][ib].first;  // nonzero column
         int d = rowA[ia][ib].second; // index of det in space 
	 dpt[ia][b] = d;
      }
   }
   // spaceA/B
   for(const auto& p : umapA) spaceA.push_back(p.first);
   for(const auto& p : umapB) spaceB.push_back(p.first);
}

// constructor
coupling_table::coupling_table(const map<onstate,int>& umap){
   auto t0 = global::get_time();
   dim = umap.size();
   C11.resize(dim);
   C22.resize(dim);
   for(const auto& pi : umap){
      for(const auto& pj : umap){
	 auto pr = pi.first.diff_type(pj.first);
	 if(pr == make_pair(1,1)){
	    C11[pi.second].push_back(pj.second);
	 }else if(pr == make_pair(2,2)){
	    C22[pi.second].push_back(pj.second);
	 }
      }
   } 
   auto t1 = global::get_time();
   cout << "coupling_table : " << dim 
	<< " time=" << global::get_duration(t1-t0) << " s" 
	<< endl; 
}

// compare with full construction
void sparse_hamiltonian::debug(const onspace& space,
			       const integral::two_body& int2e,
			       const integral::one_body& int1e){
   auto dim = connect.size();
   matrix H1(dim,dim);
   for(int i=0; i<dim; i++){
      for(int jdx=0; jdx<connect[i].size(); jdx++){
    	 int j = connect[i][jdx].first;
	 H1(i,j) = connect[i][jdx].second;
      }
   }
   auto H2 = get_Ham(space,int2e,int1e,0.0);
   for(int i=0; i<dim; i++){
      for(int j=0; j<dim; j++){
         if(abs(H1(i,j))<1.e-8 && abs(H2(i,j))<1.e-8) continue;
	 if(abs(H1(i,j)-H2(i,j))<1.e-8) continue;
	 cout << "i,j=" << i << "," << j << " "
	      << space[i] << " " << space[j]  
	      << " val=" << H1(i,j) << "," << H2(i,j) 
	      << " diff=" << H1(i,j)-H2(i,j) 
	      << " num=" << space[i].diff_num(space[j]) 
	      << endl;
      }
   } 
   cout << "|H2-H1|=" << normF(H2-H1) << endl;
}

// compute sparse H
sparse_hamiltonian::sparse_hamiltonian(const onspace& space,
				       const product_space& pspace,
				       const coupling_table& ctabA,
				       const coupling_table& ctabB,
				       const integral::two_body& int2e,
				       const integral::one_body& int1e,
				       const double ecore){
   dim = space.size();
   connect.resize(dim);
   nnz.resize(dim);
   bool debug = true;
   auto t0 = global::get_time();
 
   // 0. diagonal 
   diag = fock::get_Hdiag(space, int2e, int1e, ecore);
   auto ta = global::get_time();
   if(debug) cout << "timing for fock::get_Hdiag : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
   
   // 1. (C11+C22)_A*C00_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single/double
   // 			    {I_B,J_B} differ by zero (I_B=J_B)
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
	       double Hij = fock::get_HijS(space[i], space[j], int2e, int1e);
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
      for(int ja : ctabA.C22[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
	       double Hij = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
   } // ia
   auto tb = global::get_time();
   if(debug) cout << "timing for (C11+C22)_A*C00_B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << endl;

   // 2. C00_A*(C11+C22)_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by zero (I_A=J_A)
   // 			    {I_B,J_B} differ by single/double
   // This particular ordering of loops is optimized to minmize the
   // cache missing by exploring data locality as much as possible
   // for colB[ib], C11[ib], and dpt[ia][jb].
   for(int ib=0; ib<pspace.dimB; ib++){
      for(const auto& pa : pspace.colB[ib]){
         int ia = pa.first;
         int i = pa.second;
         for(int jb : ctabB.C11[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       double Hij = fock::get_HijS(space[i], space[j], int2e, int1e); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
         for(int jb : ctabB.C22[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       double Hij = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
   } // ib
   auto tc = global::get_time();
   if(debug) cout << "timing for C00_A*(C11+C22)_B : " << setprecision(2) 
		  << global::get_duration(tc-tb) << " s" << endl;

   // 3. C11_A*C11_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single
   // 			    {I_B,J_B} differ by single
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
         for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
            int i = pb.second;
   	    for(int jb : ctabB.C11[ib]){
   	       int j = pspace.dpt[ja][jb];
   	       if(j>i){
	          double Hij = fock::get_HijD(space[i], space[j], int2e, int1e);
	          connect[i].emplace_back(j,Hij);
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
   auto td = global::get_time();
   if(debug) cout << "timing for C11_A*C11_B : " << setprecision(2) 
   		  << global::get_duration(td-tc) << " s" << endl;
   // compute nnz
   for(int i=0; i<dim; i++){
      nnz[i] = connect[i].size();
   }
}

// matrix-vector product using stored H
void fci::get_Hx(double* y,
	         const double* x,
	         const sparse_hamiltonian& sparseH){
   // y[i] = sparseH.diag[i]*x[i]; 
   transform(sparseH.diag.begin(), sparseH.diag.end(), x, y,
	     [](const double& d, const double& c){return d*c;}); 
   // y[i] = sum_j H[i,j]*x[j] 
   for(int i=0; i<sparseH.dim; i++){
      for(int jdx=0; jdx<sparseH.nnz[i]; jdx++){
         int j = sparseH.connect[i][jdx].first;
	 double Hij = sparseH.connect[i][jdx].second;
	 assert(j > i);
	 y[i] += Hij*x[j]; // j>i
	 y[j] += Hij*x[i]; // j<i
      }
   }
}

// initial guess: v0(ndim,neig)
void fci::get_initial(const onspace& space,
		      const integral::two_body& int2e, 
		      const integral::one_body& int1e, 
	       	      const double ecore,
		      vector<double>& Diag,
		      matrix& v0){
   int ndim = v0.rows(); 
   int neig = v0.cols();
   int pdim = min(ndim, max(neig,100));
   // construct H in the low-energy subspace 
   auto index = tools::sort_index(Diag, 1);
   matrix Hpp(pdim, pdim);
   for(int j=0; j<pdim; j++){
      int jj = index[j];
      for(int i=0; i<pdim; i++){
         int ii = index[i];
	 Hpp(i,j) = fock::get_Hij(space[ii], space[jj], int2e, int1e);
      }
   }
   matrix v(Hpp);
   vector<double> e(pdim);
   eigen_solver(v, e);
   // copy back
   for(int j=0; j<neig; j++){
      for(int i=0; i<pdim; i++){
         v0(index[i],j) = v(i,j);
      }
   }
   // print
   cout << "\nfci::get_initial pdim=" << pdim << endl;
   cout << setprecision(10);
   for(int i=0; i<neig; i++){
      cout << "i=" << i << " e=" << e[i] << " " 
	   << " e+ecore=" << e[i]+ecore << endl;
   }
}

// solve eigenvalue problem in this space
void fci::ci_solver(vector<double>& es,
	       	    matrix& vs,	
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   cout << "\nfci::ci_solver dim=" << space.size() << endl; 
   bool debug = true;
   auto t0 = global::get_time();
  
   // setup product_space
   product_space pspace(space);
   auto ta = global::get_time();
   if(debug) cout << "timing for pspace : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
  
   // setupt coupling_table
   coupling_table ctabA(pspace.umapA);
   auto tb = global::get_time();
   coupling_table ctabB(pspace.umapB);
   auto tc = global::get_time();
   if(debug) cout << "timing for ctabA/B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << " "
		  << global::get_duration(tc-tb) << " s" << endl;
   
   // compute sparse_hamiltonian
   sparse_hamiltonian sparseH(space, pspace, ctabA, ctabB,
		   	      int2e, int1e, ecore);
   auto td = global::get_time();
   if(debug) cout << "timing for sparseH : " << setprecision(2) 
		  << global::get_duration(td-tc) << " s" << endl;

   // Davidson solver 
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.ndim = space.size();
   solver.neig = es.size();
   solver.Diag = sparseH.diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(fci::get_Hx, _1, _2, cref(sparseH));
   
   // get initial guess
   matrix v0(solver.ndim, solver.neig);
   get_initial(space, int2e, int1e, ecore, sparseH.diag, v0);
   auto te = global::get_time();
   if(debug) cout << "timing for get_initial : " << setprecision(2) 
		  << global::get_duration(te-td) << " s" << endl;

   // solve
   solver.solve_iter(es.data(), vs.data(), v0.data());
   //solver.solve_diag(es.data(), vs.data());
   auto tf = global::get_time();
   if(debug) cout << "timing for solve_iter : " << setprecision(2) 
		  << global::get_duration(tf-te) << " s" << endl;

   /*
   ifrdm = true;
   if(ifrdm){
      // rdm2
      int k = int1e.sorb;
      int k2 = k*(k-1)/2;
      matrix rdm2(k2,k2);
      for(int i=0; i<vs.cols(); i++){
	 vector<double> vi(vs.col(i),vs.col(i)+vs.rows());
         get_rdm12(space,vi,vi,rdm2);
	 double etot = fock::get_energy(rdm2,int2e,int1e,ecore);
	 cout << " i=" << i << " etot(RDM)=" << etot << endl;
      }
   }
   */

   cout << "timing for fci::ci_solver : " << setprecision(2) 
	<< global::get_duration(tf-t0) << " s" << endl;
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
   for(size_t i=0; i<space.size(); i++){
      // c1[i]<Di|p^+q|Di>c2[i]
      vector<int> olst;
      space[i].get_occ(olst);
      for(int p : olst){
         rdm1(p,p) += civec1[i]*civec2[i];
      }
   }
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
   for(size_t i=0; i<space.size(); i++){
      // c1[i]<Di|p0^+p1^+p1p0|Di>c2[i]
      vector<int> olst;
      space[i].get_occ(olst);
      for(int idx=0; idx<olst.size(); idx++){
         auto p0 = olst[idx]; 
	 for(int jdx=0; jdx<idx; jdx++){
            auto p1 = olst[jdx];
	    auto p01 = tools::canonical_pair0(p0,p1);
	    rdm2(p01,p01) += civec1[i]*civec2[i]; 
	 }
      }
   }
   // 1. (C11+C22)_A*C00_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single/double
   // 			    {I_B,J_B} differ by zero (I_B=J_B)
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
               int p[1], q[1];
               space[i].diff_orb(space[j],p,q);
	       auto p0 = p[0];
	       auto q0 = q[0];
               auto sgn0 = space[i].parity(p0)*space[j].parity(q0);
	       vector<int> olst;
	       space[i].get_occ(olst);
	       for(int idx=0; idx<olst.size(); idx++){
                  auto p1 = olst[idx];
	          if(p1 == p0) continue; 
                  auto sgn = sgn0;
	          auto p01 = tools::canonical_pair0(p0,p1);
	          if(p0 < p1) sgn *= -1; // sign coming from ordering of operators
	          // p1 must be not identical to q0, otherwise it cannot be in olst 
	          auto q01 = tools::canonical_pair0(q0,p1);
	          if(q0 < p1) sgn *= -1; 
                  rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
                  rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	       }
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
	       int p[2], q[2];
	       space[i].diff_orb(space[j],p,q);
	       auto p01 = tools::canonical_pair0(p[0],p[1]);
	       auto q01 = tools::canonical_pair0(q[0],q[1]);
               auto sgn = space[i].parity(p[0])*space[i].parity(p[1])
	                * space[j].parity(q[0])*space[j].parity(q[1]);
               rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
               rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	    }
	 }
      }
   } // ia
   // 2. C00_A*(C11+C22)_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by zero (I_A=J_A)
   // 			    {I_B,J_B} differ by single/double
   for(int ib=0; ib<pspace.dimB; ib++){
      for(const auto& pa : pspace.colB[ib]){
         int ia = pa.first;
         int i = pa.second;
	 // BSSB: S={A,B}
         for(int jb : ctabB.C11[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
               int p[1], q[1];
               space[i].diff_orb(space[j],p,q);
	       auto p0 = p[0];
	       auto q0 = q[0];
               auto sgn0 = space[i].parity(p0)*space[j].parity(q0);
	       vector<int> olst;
	       space[i].get_occ(olst);
	       for(int idx=0; idx<olst.size(); idx++){
                  auto p1 = olst[idx];
	          if(p1 == p0) continue; 
                  auto sgn = sgn0;
	          auto p01 = tools::canonical_pair0(p0,p1);
	          if(p0 < p1) sgn *= -1; // sign coming from ordering of operators
	          // p1 must be not identical to q0, otherwise it cannot be in olst 
	          auto q01 = tools::canonical_pair0(q0,p1);
	          if(q0 < p1) sgn *= -1; 
                  rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
                  rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	       }
	    }
	 }
	 // BBBB
         for(int jb : ctabB.C22[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       int p[2], q[2];
	       space[i].diff_orb(space[j],p,q);
	       auto p01 = tools::canonical_pair0(p[0],p[1]);
	       auto q01 = tools::canonical_pair0(q[0],q[1]);
               auto sgn = space[i].parity(p[0])*space[i].parity(p[1])
	                * space[j].parity(q[0])*space[j].parity(q[1]);
               rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
               rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
	    }
	 }
      }
   } // ib
   // 3. C11_A*C11_B: 
   // <I_A,I_B|pA^+pB^+qBqA|J_A,J_B> where {I_A,J_A} differ by single
   // 			   		   {I_B,J_B} differ by single
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
         for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
            int i = pb.second;
   	    for(int jb : ctabB.C11[ib]){
   	       int j = pspace.dpt[ja][jb];
   	       if(j>i){
	          int p[2], q[2];
	          space[i].diff_orb(space[j],p,q);
	          auto p01 = tools::canonical_pair0(p[0],p[1]);
	          auto q01 = tools::canonical_pair0(q[0],q[1]);
                  auto sgn = space[i].parity(p[0])*space[i].parity(p[1])
	                   * space[j].parity(q[0])*space[j].parity(q[1]);
                  rdm2(p01,q01) += sgn*civec1[i]*civec2[j];
                  rdm2(q01,p01) += sgn*civec1[j]*civec2[i];
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
