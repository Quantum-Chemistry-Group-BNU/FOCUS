#ifndef FCI_H
#define FCI_H

#include "fci_util.h"

namespace fci{

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
         int pdim = std::min(ndim, std::max(neig,100));
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
         std::vector<double> e(pdim);
         linalg::matrix<Tm> v;
         linalg::eig_solver(Hpp, e, v);
         // copy back
         for(int j=0; j<neig; j++){
            for(int i=0; i<pdim; i++){
               v0(index[i],j) = v(i,j);
            }
         }
         // print
         std::cout << "\nfci::get_initial pdim=" << pdim << std::endl;
         std::cout << std::setprecision(12);
         for(int i=0; i<neig; i++){
            std::cout << "i=" << i 
               << " d=" << Diag[index[i]] 
               << " e=" << e[i]+ecore << std::endl;
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
         const bool debug = true;
         const bool Htype = tools::is_complex<Tm>();
         auto t0 = tools::get_time();
         std::cout << "\nfci::ci_solver dim=" << space.size() << " Htype=" << Htype << std::endl; 
         // dimensionality check
         if(es.size() > space.size()){
            std::string msg = "error: too much roots are required! nroots,ndim=";
            tools::exit(msg+std::to_string(es.size())+","+std::to_string(space.size()));
         }
         // compute sparse_hamiltonian
         sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype);
         auto td = tools::get_time();
         // Davidson solver 
         linalg::dvdsonSolver<Tm> solver;
         solver.iprt = 1;
         solver.ndim = space.size();
         solver.neig = es.size();
         solver.Diag = sparseH.diag.data();
         using std::placeholders::_1;
         using std::placeholders::_2;
         solver.HVec = std::bind(&fci::get_Hx<Tm>, _1, _2, cref(sparseH));
         // get initial guess
         linalg::matrix<Tm> v0(solver.ndim, solver.neig);
         get_initial(space, int2e, int1e, ecore, sparseH.diag, v0);
         auto te = tools::get_time();
         if(debug) tools::timing("get_initial", td, te);
         // solve
         solver.solve_iter(es.data(), vs.data(), v0.data());
         //solver.solve_diag(es.data(), vs.data());
         auto tf = tools::get_time();
         if(debug) tools::timing("solve_iter", te, tf);
         auto t1 = tools::get_time();
         tools::timing("fci::ci_solver", t0, t1);
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

} // fci

#endif
