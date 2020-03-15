#include <functional>
#include <algorithm>
#include "../settings/global.h"
#include "../core/dvdson.h"
#include "../core/linalg.h"
#include "../core/tools.h"
#include "../core/hamiltonian.h"
#include "../core/analysis.h"
#include "fci.h"
#include "fci_rdm.h"

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
   // dimA,dimB,dpt
   dimA = udxA;
   dimB = udxB;
   dpt.resize(dimA); 
   for(int ia=0; ia<dimA; ia++){
      dpt[ia].resize(dimB,-1);
      for(const auto& pr : rowA[ia]){
	 int b = pr.first;  // nonzero column
         int d = pr.second; // // index of det in space	 
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
      for(const auto& pr : connect[i]){
         int j = get<0>(pr);	      
	 H1(i,j) = get<1>(pr);
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
	       auto pr = fock::get_HijS(space[i], space[j], int2e, int1e);
	       connect[i].push_back(make_tuple(j, pr.first, pr.second));
	    }
	 }
      }
      for(int ja : ctabA.C22[ia]){
	 for(const auto& pb : pspace.rowA[ia]){
	    int ib = pb.first;
	    int i = pb.second;
	    int j = pspace.dpt[ja][ib];
	    if(j>i){
	       auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].push_back(make_tuple(j, pr.first, pr.second));
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
	       auto pr = fock::get_HijS(space[i], space[j], int2e, int1e); 
	       connect[i].push_back(make_tuple(j, pr.first, pr.second));
	    }
	 }
         for(int jb : ctabB.C22[ib]){
	    int j = pspace.dpt[ia][jb];
	    if(j>i){
	       auto pr = fock::get_HijD(space[i], space[j], int2e, int1e); 
	       connect[i].push_back(make_tuple(j, pr.first, pr.second));
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
	          auto pr = fock::get_HijD(space[i], space[j], int2e, int1e);
	          connect[i].push_back(make_tuple(j, pr.first, pr.second));
	       } // j>0
	    } // jb
	 } // ib
      } // ja
   } // ia
   auto td = global::get_time();
   if(debug) cout << "timing for C11_A*C11_B : " << setprecision(2) 
   		  << global::get_duration(td-tc) << " s" << endl;
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
      for(const auto& pj : sparseH.connect[i]){
	 int j = get<0>(pj);
	 double Hij = get<1>(pj);
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

   // make_rdm2
   bool if_make_rdm2 = true;
   if(if_make_rdm2){
      int k = int1e.sorb;
      int k2 = k*(k-1)/2;
      cout << defaultfloat << setprecision(12);
      for(int i=0; i<vs.cols(); i++){
         matrix rdm2(k2,k2);
	 vector<double> vi(vs.col(i),vs.col(i)+vs.rows());
         make_rdm2(space,sparseH,vi,vi,rdm2);
	 double etot = fock::get_etot(rdm2,int2e,int1e,ecore);
	 assert(abs(etot-es[i]) < 1.e-8);
	 cout << " i=" << i << " etot(rdm)=" << etot << endl;
      } // i
      auto tg = global::get_time();
      if(debug) cout << "timing for make_rdm2: " << setprecision(2) 
		     << global::get_duration(tg-tf) << " s" << endl;
   }

   auto t1 = global::get_time();
   cout << "timing for fci::ci_solver : " << setprecision(2) 
	<< global::get_duration(t1-t0) << " s" << endl;
}
