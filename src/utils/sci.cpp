#include <functional>
#include <algorithm>
#include "../settings/global.h"
#include "../core/dvdson.h"
#include "../core/hamiltonian.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "sci.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace sci;

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
	 usetA.push_back(strA);
	 dsetA.resize(udxA); // make space
	 bsetA.resize(udxA);
      };
      // odd 
      auto strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 usetB.push_back(strB);
	 dsetB.resize(udxB);
	 asetB.resize(udxB);
      }
      dsetA[itA->second].push_back(i);
      bsetA[itA->second].push_back(itB->second);
      dsetB[itB->second].push_back(i);
      asetB[itB->second].push_back(itA->second);
   }
   // nnzA,nnzB
   dimA = udxA;
   nnzA.resize(dimA);
   for(int ia=0; ia<dimA; ia++) nnzA[ia] = dsetA[ia].size();
   dimB = udxB;
   nnzB.resize(dimB);
   for(int ib=0; ib<dimB; ib++) nnzB[ib] = dsetB[ib].size();	  
   // dpt
   dpt.resize(dimA); 
   for(int ia=0; ia<dimA; ia++){
      dpt[ia].resize(dimB,-1);
      for(int ib=0; ib<nnzA[ia]; ib++){
	 int b = bsetA[ia][ib]; // nonzero column
         int d = dsetA[ia][ib]; // index of det in space 
	 dpt[ia][b] = d;
      }
   }
}

// constructor
coupling_table::coupling_table(const onspace& uset){
   dim = uset.size();
   C11.resize(dim);
   C22.resize(dim);
   for(int i=0; i<dim; i++){
      for(int j=0; j<dim; j++){
	 auto pr = uset[i].diff_type(uset[j]);
	 if(pr == make_pair(1,1)){
	    C11[i].push_back(j); 
	 }else if(pr == make_pair(2,2)){
	    C22[i].push_back(j);
	 }
      }
   }
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

// C00_A*C00_B
void sparse_hamiltonian::get_C00_C00(const onspace& space, 
				     const integral::two_body& int2e, 
				     const integral::one_body& int1e, 
				     const double ecore){
   for(int i=0; i<dim; i++){
      double Hii = fock::get_Hii(space[i], int2e, int1e) + ecore; 
      connect[i].emplace_back(i,Hii);
   }
}

// (C11+C22)_A*C00_B
void sparse_hamiltonian::get_C11C22_C00(const onspace& space,
				        const product_space& pspace,
				        const coupling_table& ctabA,
				        const integral::two_body& int2e,
				        const integral::one_body& int1e){
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single/double
   // 			    {I_B,J_B} differ by zero (I_B=J_B)
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
	 for(int ib : pspace.bsetA[ia]){
	    int j = pspace.dpt[ja][ib];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       double Hij = fock::get_HijS(space[i], space[j], int2e, int1e, 2); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
      for(int ja : ctabA.C22[ia]){
	 for(int ib : pspace.bsetA[ia]){
	    int j = pspace.dpt[ja][ib];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       double Hij = fock::get_HijD(space[i], space[j], int2e, int1e, 2); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
   } // ia
}

// C00_A*(C11+C22)_B
void sparse_hamiltonian::get_C00_C11C22(const onspace& space,
				        const product_space& pspace,
				        const coupling_table& ctabB,
				        const integral::two_body& int2e,
				        const integral::one_body& int1e){
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by zero (I_A=J_A)
   // 			    {I_B,J_B} differ by single/double
   for(int ib=0; ib<pspace.dimB; ib++){
      for(int jb : ctabB.C11[ib]){
	 for(int ia : pspace.asetB[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       double Hij = fock::get_HijS(space[i], space[j], int2e, int1e, 2); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
      for(int jb : ctabB.C22[ib]){
	 for(int ia : pspace.asetB[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>=0){
	       int i = pspace.dpt[ia][ib];
	       double Hij = fock::get_HijD(space[i], space[j], int2e, int1e, 2); 
	       connect[i].emplace_back(j,Hij);
	    }
	 }
      }
   } // ib
}

// C11_A*C11_B
void sparse_hamiltonian::get_C11_C11(const onspace& space,
				     const product_space& pspace,
				     const coupling_table& ctabA,
				     const coupling_table& ctabB,
				     const integral::two_body& int2e,
				     const integral::one_body& int1e){
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single
   // 			    {I_B,J_B} differ by single
   for(int ia=0; ia<pspace.dimA; ia++){
      for(int ja : ctabA.C11[ia]){
         for(int ib : pspace.bsetA[ia]){
            int i = pspace.dpt[ia][ib];	      
   	    for(int jb : ctabB.C11[ib]){
   	       int j = pspace.dpt[ja][jb];
   	       if(j>=0){
	          double Hij = get_HijD(space[i], space[j], int2e, int1e, 2);
	          connect[i].emplace_back(j,Hij);
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
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
  
   // C00_A*C00_B
   auto t0 = global::get_time();
   this->get_C00_C00(space, int2e, int1e, ecore);
   auto ta = global::get_time();
   if(debug) cout << "timing for get_C00_C00 : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;

   // (C11+C22)_A*C00_B
   this->get_C11C22_C00(space, pspace, ctabA, int2e, int1e);
   auto tb = global::get_time();
   if(debug) cout << "timing for get_C11C22_C00 : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << endl;
   
   // C00_A*(C11+C22)_B
   this->get_C00_C11C22(space,pspace, ctabB, int2e, int1e);
   auto tc = global::get_time();
   if(debug) cout << "timing for get_C00_C11C22 : " << setprecision(2) 
		  << global::get_duration(tc-tb) << " s" << endl;

   // C11_A*C11_B  
   this->get_C11_C11(space, pspace, ctabA, ctabB, int2e, int1e);
   auto td = global::get_time();
   if(debug) cout << "timing for get_C11_C11 : " << setprecision(2) 
   		  << global::get_duration(td-tc) << " s" << endl;

   // compute nnz
   for(int i=0; i<dim; i++){
      nnz[i] = connect[i].size();
   }
}

// matrix-vector product using stored H
void sci::get_Hx(double* y,
	         const double* x,
	         const sparse_hamiltonian& sparseH){
   // y[i] = sum_j H[i,j]*x[j] 
   for(int i=0; i<sparseH.dim; i++){
      y[i] = 0.0;
      for(int jdx=0; jdx<sparseH.nnz[i]; jdx++){
         int j = sparseH.connect[i][jdx].first;
	 double Hij = sparseH.connect[i][jdx].second;
	 y[i] += Hij*x[j];
      }
   }
}

// v0(ndim,neig)
void sci::get_initial(const onspace& space,
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
   cout << "\nsci::get_initial pdim=" << pdim << endl;
   cout << setprecision(10);
   for(int i=0; i<neig; i++){
      cout << "i=" << i << " e=" << e[i] << " " 
	   << " e+ecore=" << e[i]+ecore << endl;
   }
}

// solve eigenvalue problem in this space
void sci::ci_solver(vector<double>& es,
	       	    matrix& vs,	
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   cout << "\nsci::ci_solver dim=" << space.size() << endl; 
   bool debug = true;
  
   // setup product_space
   auto t0 = global::get_time();
   product_space pspace(space);
   auto ta = global::get_time();
   if(debug) cout << "timing for pspace : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
  
   // setupt coupling_table
   coupling_table ctabA(pspace.usetA);
   auto tb = global::get_time();
   coupling_table ctabB(pspace.usetB);
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
   auto Diag = fock::get_Hdiag(space, int2e, int1e, ecore);
   solver.Diag = Diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(sci::get_Hx, _1, _2, cref(sparseH));
   
   // get initial guess
   matrix v0(solver.ndim, solver.neig);
   get_initial(space, int2e, int1e, ecore, Diag, v0);

   // solve
   solver.solve_iter(es.data(), vs.data(), v0.data());
   //solver.solve_diag(es.data(), vs.data());
   
   auto t1 = global::get_time();
   cout << "timing for sci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
