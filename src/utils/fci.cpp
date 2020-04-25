#include <functional>
#include <algorithm>
#include "../settings/global.h"
#include "../core/dvdson.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../core/analysis.h"
#include "fci.h"

using namespace std;
using namespace fock;
using namespace linalg;
using namespace fci;

// constructor
void product_space::get_pspace(const onspace& space,
		 	       const int istart){
   dimA0 = spaceA.size();
   dimB0 = spaceB.size();
   int udxA = dimA0, udxB = dimB0;
   // construct {U(A), D(A), B(A)}, {U(B), D(B), A(B)}
   int dim = space.size();
   for(int i=istart; i<dim; i++){
      // even bits 
      onstate strA = space[i].get_even();
      auto itA = umapA.find(strA);
      if(itA == umapA.end()){
	 spaceA.push_back(strA);
         auto pr = umapA.insert({strA,udxA});
	 itA = pr.first;
	 udxA += 1;
	 rowA.resize(udxA); // reserve additional space for the new row
      };
      // odd bits
      onstate strB = space[i].get_odd();
      auto itB = umapB.find(strB);
      if(itB == umapB.end()){
	 spaceB.push_back(strB);
         auto pr = umapB.insert({strB,udxB});
	 itB = pr.first;
	 udxB += 1;
	 colB.resize(udxB);
      }
      rowA[itA->second].emplace_back(itB->second,i);
      colB[itB->second].emplace_back(itA->second,i);
   }
   assert(udxA == spaceA.size());
   assert(udxB == spaceB.size());
   dimA = udxA;
   dimB = udxB;
}

// coupling_table
void coupling_table::get_C11(const onspace& space,
			     const int istart){
   auto t0 = global::get_time();
   int avg = 0;
   int dim = space.size();
   C11.resize(dim);
   for(int i=0; i<dim; i++){
      if(i<istart){
         for(int j=istart; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == make_pair(1,1)) C11[i].insert(j);
         }
      }else{
         for(int j=0; j<dim; j++){
            auto pr = space[i].diff_type(space[j]);
            if(pr == make_pair(1,1)) C11[i].insert(j);
         }
      }
      avg += C11[i].size();
   }
   auto t1 = global::get_time();
   cout << "coupling_table::get_C11 : dim=" << dim
	<< " avg=" << avg/dim
	<< " timing : " << global::get_duration(t1-t0) << " s" << endl; 
}

// compare with full construction
void sparse_hamiltonian::debug(const onspace& space,
			       const integral::two_body& int2e,
			       const integral::one_body& int1e){
   auto dim = connect.size();
   matrix H1(dim,dim);
   for(int i=0; i<dim; i++){
      for(int jdx=0; jdx<connect[i].size(); jdx++){
	 int j = connect[i][jdx];
	 H1(i,j) = value[i][jdx];
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
void sparse_hamiltonian::get_hamiltonian(const onspace& space,
				         const product_space& pspace,
				         const coupling_table& ctabA,
				         const coupling_table& ctabB,
				         const integral::two_body& int2e,
				         const integral::one_body& int1e,
				         const double ecore,
					 const int istart){
   cout << "\nsparse_hamiltonian::get_hamiltonian" 
	<< " dim0 = " << istart << " dim = " << space.size() << endl; 
   bool debug = true;
   auto t0 = global::get_time();
   // initialization for the first use
   if(istart == 0){
      diag.clear();
      connect.clear();
      value.clear();
      diff.clear();
   }
   // diagonal 
   dim = space.size();
   diag.resize(dim);
   for(size_t i=istart; i<dim; i++){
      diag[i] = fock::get_Hii(space[i],int2e,int1e) + ecore;
   }
   auto ta = global::get_time();
   if(debug) cout << "timing for diagonal : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
   // off-diagonal 
   connect.resize(dim);
   value.resize(dim);
   diff.resize(dim);
   // 1. (C11+C22)_A*C00_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single/double
   // 	 		    {I_B,J_B} differ by zero (I_B=J_B)
   for(int ia=0; ia<pspace.dimA; ia++){
      for(const auto& pib : pspace.rowA[ia]){
	 int ib = pib.first;
	 int i = pib.second;
	 if(i < istart) continue; // incremental build
	 for(const auto& pja : pspace.colB[ib]){
	    int ja = pja.first;
	    int j = pja.second;
	    if(j >= i) continue; 
	    // check connectivity <I_A|H|J_A>
	    auto pr = pspace.spaceA[ia].diff_type(pspace.spaceA[ja]);
	    if(pr == make_pair(1,1)){
	       auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
	       connect[i].push_back(j);
	       value[i].push_back(pr.first);
	       diff[i].push_back(pr.second);
	    }else if(pr == make_pair(2,2)){
	       auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].push_back(j);
	       value[i].push_back(pr.first);
	       diff[i].push_back(pr.second);
	    }
	 } // ja
      } // ib
   } // ia
   auto tb = global::get_time();
   if(debug) cout << "timing for (C11+C22)_A*C00_B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << endl;
   // 2. C00_A*(C11+C22)_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by zero (I_A=J_A)
   // 			    {I_B,J_B} differ by single/double
   for(int ia=0; ia<pspace.dimA; ia++){
      for(const auto& pib : pspace.rowA[ia]){
	 int ib = pib.first;
	 int i = pib.second;
	 if(i < istart) continue; // incremental build
         for(const auto& pjb : pspace.rowA[ia]){
	    int jb = pjb.first;
	    int j = pjb.second;
	    if(j >= i) continue; 
	    // check connectivity <I_B|H|J_B>
	    auto pr = pspace.spaceB[ib].diff_type(pspace.spaceB[jb]);
	    if(pr == make_pair(1,1)){
	       auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
	       connect[i].push_back(j);
	       value[i].push_back(pr.first);
	       diff[i].push_back(pr.second);
	    }else if(pr == make_pair(2,2)){
	       auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].push_back(j);
	       value[i].push_back(pr.first);
	       diff[i].push_back(pr.second);
	    }
	 } // jb
      } // ib
   } // ia
   auto tc = global::get_time();
   if(debug) cout << "timing for C00_A*(C11+C22)_B : " << setprecision(2) 
		  << global::get_duration(tc-tb) << " s" << endl;
   // 3. C11_A*C11_B:
   // <I_A,I_B|H|J_A,J_B> = {I_A,J_A} differ by single
   // 			    {I_B,J_B} differ by single
   for(int ia=0; ia<pspace.dimA; ia++){
      for(const auto& pib : pspace.rowA[ia]){
	 int ib = pib.first;
	 int i = pib.second;
	 if(i < istart) continue; // incremental build
	 for(int ja : ctabA.C11[ia]){
	    for(const auto& pjb : pspace.rowA[ja]){
	       int jb = pjb.first;
	       int j = pjb.second;	       
   	       if(j >=i) continue;
	       auto search = ctabB.C11[ib].find(jb);
	       if(search != ctabB.C11[ib].end()){
	          auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
	          connect[i].push_back(j);
	          value[i].push_back(pr.first);
	          diff[i].push_back(pr.second);
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
   auto td = global::get_time();
   if(debug) cout << "timing for C11_A*C11_B : " << setprecision(2) 
   		  << global::get_duration(td-tc) << " s" << endl;
   auto t1 = global::get_time();
   cout << "timing for sparse_hamiltonian::get_hamiltonian : " 
	<< setprecision(2) << global::get_duration(t1-t0) << " s" << endl;
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
      for(int jdx=0; jdx<sparseH.connect[i].size(); jdx++){
	 int j = sparseH.connect[i][jdx];
	 double Hij = sparseH.value[i][jdx];
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
   cout << setprecision(12);
   for(int i=0; i<neig; i++){
      cout << "i=" << i << " e=" << e[i] << " " 
	   << " e+ecore=" << e[i]+ecore << endl;
   }
}

// solve eigenvalue problem in this space
void fci::ci_solver(sparse_hamiltonian& sparseH,	    
		    vector<double>& es,
	       	    matrix& vs,
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   cout << "\nfci::ci_solver dim=" << space.size() << endl; 
   bool debug = true;
   auto t0 = global::get_time();
   
   // dimensionality check
   if(es.size() > space.size()){
      cout << "error: too much roots are required! nroot,ndim=" 
	   << es.size() << "," << space.size() << endl;
      exit(1);
   }
  
   // setup product_space
   product_space pspace;
   pspace.get_pspace(space);
   auto ta = global::get_time();
   if(debug) cout << "timing for pspace : " << setprecision(2) 
		  << global::get_duration(ta-t0) << " s" << endl;
  
   // setupt coupling_table
   coupling_table ctabA, ctabB;
   ctabA.get_C11(pspace.spaceA);
   auto tb = global::get_time();
   ctabB.get_C11(pspace.spaceB);
   auto tc = global::get_time();
   if(debug) cout << "timing for ctabA/B : " << setprecision(2) 
		  << global::get_duration(tb-ta) << " s" << " "
		  << global::get_duration(tc-tb) << " s" << endl;
   
   // compute sparse_hamiltonian
   sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
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

   auto t1 = global::get_time();
   cout << "timing for fci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}

// without sparseH as output
void fci::ci_solver(vector<double>& es,
	       	    matrix& vs,
		    const onspace& space,
	       	    const integral::two_body& int2e,
	       	    const integral::one_body& int1e,
	       	    const double ecore){
   sparse_hamiltonian sparseH;
   ci_solver(sparseH, es, vs, space, int2e, int1e, ecore);
}

// compute S & H
matrix fci::get_Smat(const onspace& space,
 		     const vector<vector<double>>& vs){
   int dim = space.size();
   int n = vs.size();
   matrix Smat(n,n);
   for(int j=0; j<n; j++){
      for(int i=0; i<n; i++){
   	 // SIJ = <I|S|J>
	 Smat(i,j) = ddot(dim,vs[i].data(),vs[j].data());
      }
   }
   return Smat;
}

matrix fci::get_Hmat(const onspace& space,
 		     const vector<vector<double>>& vs,
	       	     const integral::two_body& int2e,
	       	     const integral::one_body& int1e,
	             const double ecore){
   // setup product_space
   product_space pspace;
   pspace.get_pspace(space);
   // setupt coupling_table
   coupling_table ctabA, ctabB;
   ctabA.get_C11(pspace.spaceA);
   ctabB.get_C11(pspace.spaceB);
   // compute sparse_hamiltonian
   sparse_hamiltonian sparseH;
   sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
		   	   int2e, int1e, ecore);
   int dim = space.size();
   int n = vs.size();
   matrix Hmat(n,n);
   for(int j=0; j<n; j++){
      vector<double> Hx(dim,0.0);
      fci::get_Hx(Hx.data(),vs[j].data(),sparseH);
      for(int i=0; i<n; i++){
         // HIJ = <I|H|J>
	 Hmat(i,j) = ddot(dim,vs[i].data(),Hx.data());
      }
   }
   return Hmat;
}
