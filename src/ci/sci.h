#ifndef SCI_H
#define SCI_H

#include "sci_util.h"
#include "fci.h"

namespace sci{

// selected CI procedure
template <typename Tm>
void ci_solver(const input::schedule& schd,
	       fci::sparse_hamiltonian<Tm>& sparseH,
	       std::vector<double>& es,
	       std::vector<std::vector<Tm>>& vs,
	       fock::onspace& space,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const double ecore){
   const bool Htype = fci::is_complex<Tm>();
   bool debug = false;
   auto t0 = tools::get_time();
   std::cout << "\nsci::ci_solver Htype=" << Htype << std::endl; 
   // set up head-bath table
   heatbath_table hbtab(int2e, int1e);
   // set up intial configurations
   std::vector<double> esol;
   linalg::matrix<Tm> vsol;
   std::unordered_set<fock::onstate> varSpace;
   get_initial(esol, vsol, space, varSpace, 
	       hbtab, schd, int2e, int1e, ecore);
   // set up auxilliary data structure   
   sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype);
   // start increment
   bool ifconv = false;
   int nsub = space.size(); 
   int neig = schd.nroots;
   for(int iter=0; iter<schd.maxiter; iter++){
      std::cout << "\n---------------------" << std::endl;
      std::cout << "iter=" << iter << " eps1=" << schd.eps1[iter] << std::endl;
      std::cout << "---------------------" << std::endl;
      double eps1 = schd.eps1[iter];
      // compute cmax[i] = \sqrt{\sum_j|vj[i]|^2/n} for screening
      std::vector<double> cmax(nsub,0.0);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub; i++){
	    cmax[i] += std::norm(vsol(i,j));
         }
      }
      std::transform(cmax.begin(), cmax.end(), cmax.begin(),
		     [neig](const double& x){ return pow(x/neig,0.5); });
      // expand 
      expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.flip);
      int nsub0 = nsub;
      nsub = space.size();
      // update auxilliary data structure 
      sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype, nsub0);
      // set up Davidson solver 
      linalg::dvdsonSolver<Tm> solver;
      solver.iprt = 1;
      solver.crit_v = schd.crit_v;
      solver.maxcycle = schd.maxcycle;
      solver.ndim = nsub;
      solver.neig = neig;
      solver.Diag = sparseH.diag.data();
      using std::placeholders::_1;
      using std::placeholders::_2;
      solver.HVec = std::bind(&fci::get_Hx<Tm>, _1, _2, std::cref(sparseH));
      // copy previous initial guess
      linalg::matrix<Tm> v0(nsub, neig);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub0; i++){
            v0(i,j) = vsol(i,j);
	 }
      }
      // solve
      std::cout << std::endl;
      std::vector<double> esol1(neig);
      linalg::matrix<Tm> vsol1(nsub, neig);
      solver.solve_iter(esol1.data(), vsol1.data(), v0.data());

      // check convergence of SCI
      std::vector<bool> conv(neig);
      std::cout << std::endl;
      for(int i=0; i<neig; i++){
	 conv[i] = abs(esol1[i]-esol[i]) < schd.deltaE; 
	 std::vector<Tm> vtmp(vsol1.col(i),vsol1.col(i)+nsub);
         double SvN = fock::coeff_entropy(vtmp); 
	 std::cout << "sci: iter=" << iter
	      << " eps1=" << std::scientific << std::setprecision(2) << schd.eps1[iter]
	      << " nsub=" << nsub 
	      << " i=" << i 
	      << " e=" << std::defaultfloat << std::setprecision(12) << esol1[i] 
	      << " de=" << std::scientific << std::setprecision(2) << esol1[i]-esol[i] 
	      << " conv=" << conv[i] 
	      << " SvN=" << SvN
	      << std::endl;
      }
      esol = esol1;
      vsol = vsol1;
      ifconv = (count(conv.begin(), conv.end(), true) == neig);
      if(iter>=schd.miniter && ifconv){
	 std::cout << "\nsci convergence is achieved!" << std::endl;
	 break;
      }
   } // iter
   if(!ifconv){
      std::cout << "\nsci convergence failure: out of maxiter=" << schd.maxiter << std::endl;
   }
   // finally save results
   copy_n(esol.begin(), neig, es.begin());
   for(int i=0; i<neig; i++){
      vs[i].resize(nsub);
      copy_n(vsol.col(i), nsub, vs[i].begin());
   }
   auto t1 = tools::get_time();
   std::cout << "timing for sci::ci_solver : " << std::setprecision(2) 
  	     << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // sci

#endif
