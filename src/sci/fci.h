#ifndef FCI_H
#define FCI_H

#include <string>
#include <vector>
#include "../core/onspace.h"
#include "../core/matrix.h"
#include "../core/integral.h"
#include "../core/tools.h"
#include "fci_util.h"

namespace fci{

// matrix-vector product using stored H
template <typename Tm>
void get_Hx(Tm* y,
	    const Tm* x,
	    const sparse_hamiltonian<Tm>& sparseH){
   // y[i] = sparseH.diag[i]*x[i]; 
   transform(sparseH.diag.begin(), sparseH.diag.end(), x, y,
	     [](const double& d, const Tm& c){return d*c;}); 
   // y[i] = sum_j H[i,j]*x[j] 
   for(int i=0; i<sparseH.dim; i++){
      for(int jdx=0; jdx<sparseH.connect[i].size(); jdx++){
	 int j = sparseH.connect[i][jdx];
	 Tm Hij = sparseH.value[i][jdx];
	 y[i] += Hij*x[j]; // j>i
	 y[j] += tools::conjugate(Hij)*x[i]; // j<i 
      }
   }
}

// initial guess: v0(ndim,neig)
template <typename Tm>
void get_initial(const fock::onspace& space,
	         const integral::two_body<Tm>& int2e,
	         const integral::one_body<Tm>& int1e,
		 const double ecore,
		 std::vector<double>& Diag,
		 linalg::matrix<Tm>& v0){
   int ndim = v0.rows(); 
   int neig = v0.cols();
   int pdim = min(ndim, max(neig,100));
   // construct H in the low-energy subspace 
   auto index = tools::sort_index(Diag);
   linalg::matrix<Tm> Hpp(pdim, pdim);
   for(int j=0; j<pdim; j++){
      int jj = index[j];
      for(int i=0; i<pdim; i++){
         int ii = index[i];
	 Hpp(i,j) = fock::get_Hij(space[ii], space[jj], int2e, int1e);
      }
   }
   vector<double> e(pdim);
   linalg::matrix<Tm> v;
   eig_solver(Hpp, e, v);
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
      cout << "i=" << i 
	   << " d=" << Diag[index[i]] 
	   << " e=" << e[i]+ecore << endl;
   }
}

// solve eigenvalue problem in this space
template <typename Tm>
void ci_solver(sparse_hamiltonian<Tm>& sparseH,
	       std::vector<double>& es,
	       linalg::matrix<Tm>& vs,	
	       const fock::onspace& space,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore){
   bool debug = true;
   const bool Htype = is_complex<Tm>();
   auto t0 = tools::get_time();
   cout << "\nfci::ci_solver dim=" << space.size() << " Htype=" << Htype << endl; 
   // dimensionality check
   if(es.size() > space.size()){
      cout << "error: too much roots are required! nroot,ndim=" 
	   << es.size() << "," << space.size() << endl;
      exit(1);
   }
   // 1. setup product_space
   product_space pspace;
   pspace.get_pspace(space);
   auto ta = tools::get_time();
   if(debug) cout << "timing for pspace : " << setprecision(2) 
		  << tools::get_duration(ta-t0) << " s" << endl;
   // 2. setupt coupling_table
   coupling_table ctabA, ctabB;
   ctabA.get_Cmn(pspace.spaceA, Htype);
   auto tb = tools::get_time();
   ctabB.get_Cmn(pspace.spaceB, Htype);
   auto tc = tools::get_time();
   if(debug) cout << "timing for ctabA/B : " << setprecision(2) 
		  << tools::get_duration(tb-ta) << " s" << " "
		  << tools::get_duration(tc-tb) << " s" << endl; 
   // 3. compute sparse_hamiltonian
   sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
		   	   int2e, int1e, ecore, Htype);
   sparseH.check(space, int2e, int1e, ecore);
   sparseH.analysis();
   exit(1);
   auto td = tools::get_time();
   if(debug) cout << "timing for sparseH : " << setprecision(2) 
		  << tools::get_duration(td-tc) << " s" << endl;
   // 4. Davidson solver 
   linalg::dvdsonSolver<Tm> solver;
   solver.iprt = 1;
   solver.ndim = space.size();
   solver.neig = es.size();
   solver.Diag = sparseH.diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(&fci::get_Hx<Tm>, _1, _2, cref(sparseH));
   // get initial guess
   linalg::matrix<Tm> v0(solver.ndim, solver.neig);
   get_initial(space, int2e, int1e, ecore, sparseH.diag, v0);
   auto te = tools::get_time();
   if(debug) cout << "timing for get_initial : " << setprecision(2) 
		  << tools::get_duration(te-td) << " s" << endl;
   // solve
   solver.solve_iter(es.data(), vs.data(), v0.data());
   //solver.solve_diag(es.data(), vs.data());
   auto tf = tools::get_time();
   if(debug) cout << "timing for solve_iter : " << setprecision(2) 
		  << tools::get_duration(tf-te) << " s" << endl;
   auto t1 = tools::get_time();
   cout << "timing for fci::ci_solver : " << setprecision(2) 
	<< tools::get_duration(t1-t0) << " s" << endl;
}

// without sparseH as output
template <typename Tm>
void ci_solver(std::vector<double>& es,
	       linalg::matrix<Tm>& vs,	
	       const fock::onspace& space,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore){
   sparse_hamiltonian<Tm> sparseH;
   ci_solver(sparseH, es, vs, space, int2e, int1e, ecore);
}

// compute S & H
template <typename Tm>
linalg::matrix<Tm> get_Smat(const fock::onspace& space,
 		            const std::vector<std::vector<Tm>>& vs){
   int dim = space.size();
   int n = vs.size();
   linalg::matrix<Tm> Smat(n,n);
   for(int j=0; j<n; j++){
      for(int i=0; i<n; i++){
   	 // SIJ = <I|S|J>
	 Smat(i,j) = xdot(dim,vs[i].data(),vs[j].data());
      }
   }
   return Smat;
}

template <typename Tm>
linalg::matrix<Tm> get_Hmat(const fock::onspace& space,
 		            const std::vector<std::vector<Tm>>& vs,
	       	            const integral::two_body<Tm>& int2e,
	       	            const integral::one_body<Tm>& int1e,
	                    const double ecore){
   const bool Htype = is_complex<Tm>();
   // setup product_space
   product_space pspace;
   pspace.get_pspace(space);
   // setupt coupling_table
   coupling_table ctabA, ctabB;
   ctabA.get_Cmn(pspace.spaceA, Htype);
   ctabB.get_Cmn(pspace.spaceB, Htype);
   // compute sparse_hamiltonian
   sparse_hamiltonian<Tm> sparseH;
   sparseH.get_hamiltonian(space, pspace, ctabA, ctabB,
		   	   int2e, int1e, ecore, Htype);
   int dim = space.size();
   int n = vs.size();
   linalg::matrix<Tm> Hmat(n,n);
   for(int j=0; j<n; j++){
      std::vector<Tm> Hx(dim,0.0);
      fci::get_Hx(Hx.data(),vs[j].data(),sparseH);
      for(int i=0; i<n; i++){
         // HIJ = <I|H|J>
	 Hmat(i,j) = xdot(dim,vs[i].data(),Hx.data());
      }
   }
   return Hmat;
}

// io: save/load onspace & ci vectors
template <typename Tm>
void ci_save(const fock::onspace& space,
	     const std::vector<std::vector<Tm>>& vs,
	     const std::string fname="ci.info"){
   std::cout << "\nfci::ci_save" << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << space << vs;
}

template <typename Tm>
void ci_load(fock::onspace& space,
	     std::vector<std::vector<Tm>>& vs,
	     const std::string fname="ci.info"){
   std::cout << "\nfci::ci_load" << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> space >> vs;
}

} // fci

#endif
