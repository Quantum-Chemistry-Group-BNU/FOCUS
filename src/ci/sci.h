#ifndef SCI_H
#define SCI_H

#include "fci_util.h"
#include "sci_util.h"

namespace sci{

// prepare intial solution
template <typename Tm>
void get_initial(std::vector<double>& es,
		 linalg::matrix<Tm>& vs,
		 fock::onspace& space,
	         std::unordered_set<fock::onstate>& varSpace,
		 const heatbath_table<Tm>& hbtab, 
		 const input::schedule& schd, 
	         const integral::two_body<Tm>& int2e,
	         const integral::one_body<Tm>& int1e,
	         const double ecore){
   std::cout << "\nsci::get_initial" << std::endl;
   // space = {|Di>}
   const int k = int1e.sorb;
   int ndet = 0;
   for(const auto& det : schd.sci.det_seeds){
      // consistency check
      std::cout << ndet << "-th det: ";
      for(auto k : det) std::cout << k << " ";
      std::cout << std::endl;
      ndet += 1;
      if(det.size() != schd.nelec){
         std::cout << "det.size=" << det.size() << " schd.nelec=" << schd.nelec << std::endl;
         tools::exit("error: det.size is inconsistent with schd.nelec!");
      }
      // convert det to onstate
      fock::onstate state(k); 
      for(int i : det) state[i] = 1;
      // search first
      auto search = varSpace.find(state);
      if(search == varSpace.end()){
	 varSpace.insert(state);
	 space.push_back(state);
      }
      // flip determinant 
      if(schd.sci.flip){
         auto state1 = state.flip();
         auto search1 = varSpace.find(state1);
         if(search1 == varSpace.end()){
            space.push_back(state1);
            varSpace.insert(state1);
         }
      }
   }
   // print
   std::cout << "energies for reference states:" << std::endl;
   std::cout << std::defaultfloat << std::setprecision(12);
   int nsub = space.size();
   for(int i=0; i<nsub; i++){
      std::cout << "i = " << i << " state = " << space[i]
	   << " e = " << fock::get_Hii(space[i],int2e,int1e)+ecore 
	   << std::endl;
   }
   // selected CISD space
   double eps1 = schd.sci.eps0;
   std::vector<double> cmax(nsub,1.0);
   expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.sci.flip);
   nsub = space.size();
   // set up initial states
   if(schd.sci.nroots > nsub) tools::exit("error: subspace is too small in sci::get_initial!");
   linalg::matrix<Tm> H = fock::get_Hmat(space, int2e, int1e, ecore);
   std::vector<double> esol(nsub);
   linalg::matrix<Tm> vsol;
   linalg::eig_solver(H, esol, vsol);
   // save
   int neig = schd.sci.nroots;
   es.resize(neig);
   vs.resize(nsub, neig);
   for(int j=0; j<neig; j++){
      for(int i=0; i<nsub; i++){
	 vs(i,j) = vsol(i,j);
      }
      es[j] = esol[j];
   }
   // print
   std::cout << std::setprecision(12);
   for(int i=0; i<neig; i++){
      std::cout << "i = " << i << " e = " << es[i] << std::endl; 
   }
}

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
   const bool Htype = tools::is_complex<Tm>();
   bool debug = false;
   auto t0 = tools::get_time();
   std::cout << "\nsci::ci_solver Htype=" << Htype << std::endl; 
   // set up head-bath table
   heatbath_table<Tm> hbtab(int2e, int1e);
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
   int neig = schd.sci.nroots;
   for(int iter=0; iter<schd.sci.maxiter; iter++){
      std::cout << "\n---------------------" << std::endl;
      std::cout << "iter=" << iter << " eps1=" << std::scientific << schd.sci.eps1[iter] << std::endl;
      std::cout << "---------------------" << std::endl;
      double eps1 = schd.sci.eps1[iter];
      // compute cmax[i] = \sqrt{\sum_j|vj[i]|^2} for screening
      std::vector<double> cmax(nsub,0.0);
      for(int j=0; j<neig; j++){
         for(int i=0; i<nsub; i++){
	    cmax[i] += std::norm(vsol(i,j));
         }
      }
      std::transform(cmax.begin(), cmax.end(), cmax.begin(),
		     [neig](const double& x){ return std::pow(x,0.5); });
      // expand 
      expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.sci.flip);
      int nsub0 = nsub;
      nsub = space.size();
      // update auxilliary data structure 
      sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype, nsub0);
      // set up Davidson solver 
      linalg::dvdsonSolver<Tm> solver(nsub, neig, schd.sci.crit_v, schd.sci.maxcycle);
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
	 conv[i] = std::abs(esol1[i]-esol[i]) < schd.sci.deltaE; 
	 std::vector<Tm> vtmp(vsol1.col(i),vsol1.col(i)+nsub);
         double SvN = fock::coeff_entropy(vtmp); 
	 std::cout << "sci: iter=" << iter
	      << " eps1=" << std::scientific << std::setprecision(2) << schd.sci.eps1[iter]
	      << " nsub=" << nsub 
	      << " i=" << i 
	      << " e=" << std::defaultfloat << std::setprecision(12) << esol1[i] 
	      << " de=" << std::scientific << std::setprecision(2) << esol1[i]-esol[i] 
	      << " conv=" << conv[i] 
	      << " SvN=" << SvN
	      << std::endl;
	 fock::coeff_analysis(vtmp);
      }
      esol = esol1;
      vsol = vsol1;
      ifconv = (count(conv.begin(), conv.end(), true) == neig);
      if(iter>=schd.sci.miniter && ifconv){
	 std::cout << "\nsci convergence is achieved!" << std::endl;
	 break;
      }
   } // iter
   if(!ifconv){
      std::cout << "\nsci convergence failure: out of maxiter=" << schd.sci.maxiter << std::endl;
   }
   // finally save results
   copy_n(esol.begin(), neig, es.begin());
   for(int i=0; i<neig; i++){
      vs[i].resize(nsub);
      copy_n(vsol.col(i), nsub, vs[i].begin());
   }
   auto t1 = tools::get_time();
   tools::timing("sci::ci_solver", t0, t1);
}

} // sci

#endif
