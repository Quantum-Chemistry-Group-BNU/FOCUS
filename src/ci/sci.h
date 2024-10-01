#ifndef SCI_H
#define SCI_H

#include "fci_util.h"
#include "sci_util.h"
#include "../core/binom.h"

namespace sci{

   // prepare intial subspace vai aufbau principle
   template <typename Tm>
      void init_aufbau(fock::onspace& space,
            std::unordered_set<fock::onstate>& varSpace,
            const input::schedule& schd, 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const bool debug=true){
         const int nroots = schd.ci.nroots;
         const int ne = schd.nelec;
         const int tm = schd.twom;
         const int k = int1e.sorb;
         const int ks = k/2;
         std::cout << "\nsci::init_aufbau (k,ne,tm)=" << k << "," << ne << "," << tm 
            << " checkms=" << schd.ci.checkms << " nroots=" << nroots << std::endl;

         // generate subspace via aufbau principle 
         int ksmin = 0, kmin = 0;
         size_t dim = 0;
         fock::onspace subspace;
         // determine (ksmin,kmin) first
         if(!schd.ci.checkms){
            for(int km=ne; km<k; km++){
               dim = fock::binom(km,ne);
               if(dim >= nroots){
                  kmin = km;
                  break;
               }
            } 
            if(kmin == 0){
               std::cout << "error: nroots required exceed the dimension of Hilbert space dim=" << dim << std::endl;
               exit(1);
            }
            if(kmin%2 == 1) kmin += 1;
            ksmin = kmin/2;
            subspace = fock::get_fci_space(kmin, ne);
         }else{
            // consider ms constraint
            int na = (ne+tm)/2;
            int nb = (ne-tm)/2;
            int ksta = std::max(std::min(ks-na,na),std::min(ks-nb,nb));
            for(int km=ksta; km<ks; km++){
               dim = fock::binom(km,na)*fock::binom(km,nb);
               if(dim >= nroots){
                  ksmin = km;
                  break;
               }
            }
            if(ksmin == 0){
               std::cout << "error: nroots required exceed the dimension of Hilbert space dim=" << dim << std::endl;
               exit(1);
            }
            kmin = ksmin*2;
            subspace = fock::get_fci_space(ksmin, na, nb);
         }
         dim = subspace.size();

         // construct initial subspace
         fock::onspace fspace(dim), mspace(nroots);
         std::vector<double> econf(dim);
         std::vector<double> enorb(ks);
         for(int iter=0; iter<2; iter++){

            // ''spatial orbital energy''
            for(int i=0; i<ks; i++){
               enorb[i] = std::real(int1e.get(2*i,2*i)+int1e.get(2*i+1,2*i+1))/2.0;
            }
            if(iter > 0){
               // take into account the electron-electron interaction approximately
               for(int j=0; j<nroots; j++){
                  std::vector<int> olst;
                  mspace[j].get_olst(olst);
                  for(int p=0; p<olst.size(); p++){
                     for(int i=0; i<ks; i++){
                        enorb[i] += std::real(int2e.getQ(2*i,olst[p])+int2e.getQ(2*i+1,olst[p]))/(2.0*nroots);
                     }
                  }
               }
            }
            auto index = tools::sort_index(enorb);
            for(int i=0; i<ks; i++){
               int idx = index[i];
               if(debug) std::cout << "i=" << i << " idx=" << idx << " enorb=" << std::setprecision(8) << enorb[idx] << std::endl;
            }

            // map subspace configuration to full space
            for(size_t i=0; i<dim; i++){
               fock::onstate state(k);
               for(int j=0; j<kmin; j++){
                  int js = j/2, spin_j = j%2;
                  if(subspace[i][j] == 1) state[2*index[js]+spin_j] = 1;
               }
               fspace[i] = state;
               econf[i] = fock::get_Hii(state, int2e, int1e) + ecore;
               if(debug) std::cout << "iter=" << iter << " i=" << i << " state=" << state << " econf=" << econf[i] << std::endl;
            }

            auto index2 = tools::sort_index(econf);
            for(int i=0; i<nroots; i++){
               mspace[i] = fspace[index2[i]];
               if(debug) std::cout << "selected: i=" << i << " state=" << mspace[i] << " econf=" << econf[index2[i]] << std::endl; 
            }
     
            // take the union of the lowest energy states
            for(int i=0; i<nroots; i++){
               const auto& state = mspace[i];
               // search first
               auto search = varSpace.find(state);
               if(search == varSpace.end()){
                  varSpace.insert(state);
                  space.push_back(state);
               }
               // flip determinant 
               if(schd.ci.flip){
                  auto state1 = state.flip();
                  auto search1 = varSpace.find(state1);
                  if(search1 == varSpace.end()){
                     space.push_back(state1);
                     varSpace.insert(state1);
                  }
               }
            }
         } // iter
      }

   // prepare intial subspace via probalistic methods
   template <typename Tm>
      void init_sampling(fock::onspace& space,
            std::unordered_set<fock::onstate>& varSpace,
            const input::schedule& schd, 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const bool debug=true){
         const int nroots = schd.ci.nroots;
         const int ne = schd.nelec;
         const int tm = schd.twom;
         const int k = int1e.sorb;
         const int ks = k/2;
         std::cout << "\nsci::init_sampling (k,ne,tm)=" << k << "," << ne << "," << tm 
            << " checkms=" << schd.ci.checkms << " nroots=" << nroots << std::endl;

         // generate subspace via aufbau principle 
         std::vector<double> enorb(ks);
         // ''spatial orbital energy''
         for(int i=0; i<ks; i++){
            enorb[i] = std::real(int1e.get(2*i,2*i)+int1e.get(2*i+1,2*i+1))/2.0;
         }
         auto index = tools::sort_index(enorb);
         for(int i=0; i<ks; i++){
            int idx = index[i];
            if(debug) std::cout << "i=" << i << " idx=" << idx << " enorb=" << std::setprecision(8) << enorb[idx] << std::endl;
         }

         exit(1);
      }

   // prepare intial solution
   template <typename Tm>
      void init_ciwf(std::vector<double>& es,
            linalg::matrix<Tm>& vs,
            fock::onspace& space,
            std::unordered_set<fock::onstate>& varSpace,
            const heatbath_table<Tm>& hbtab, 
            const input::schedule& schd, 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore){
         std::cout << "\nsci::init_ciwf schd.ci.init=" << schd.ci.init << std::endl;
         auto t0 = tools::get_time();
         
         // space = {|Di>}
         const int k = int1e.sorb;
         int ndet = 0;
         if(schd.ci.det_seeds.size() > 0){

            // generate initial subspace from input dets
            for(const auto& det : schd.ci.det_seeds){
               // consistency check
               std::cout << ndet << "-th det: ";
               for(auto iorb : det) std::cout << iorb << " ";
               std::cout << std::endl;
               ndet += 1;
               if(det.size() != schd.nelec){
                  std::cout << "det.size=" << det.size() << " schd.nelec=" << schd.nelec << std::endl;
                  tools::exit("error: det.size is inconsistent with schd.nelec!");
               }
               // convert det to onstate
               fock::onstate state(k); 
               for(int iorb : det) state[iorb] = 1;
               // check Ms value if necessary
               if(schd.ci.checkms and state.twom() != schd.twom){
                  std::cout << "error: inconsistent twom:"
                            << " twom[input]=" << schd.twom
                            << " twom[det]=" << state.twom() 
                            << " det=" << state 
                            << std::endl;
                  exit(1);
               }
               // search first
               auto search = varSpace.find(state);
               if(search == varSpace.end()){
                  varSpace.insert(state);
                  space.push_back(state);
               }
               // flip determinant 
               if(schd.ci.flip){
                  auto state1 = state.flip();
                  auto search1 = varSpace.find(state1);
                  if(search1 == varSpace.end()){
                     space.push_back(state1);
                     varSpace.insert(state1);
                  }
               }
            }

         }else{

            // generate initial determinants according to integrals
            if(schd.ci.init == "aufbau"){
               init_aufbau(space, varSpace, schd, int2e, int1e, ecore);
            }else if(schd.ci.init == "sampling"){
               init_sampling(space, varSpace, schd, int2e, int1e, ecore);
            }else{
               std::cout << "error: no such option in schd.ci.init=" << schd.ci.init << std::endl;
               exit(1);
            }

         }
         // print
         std::cout << "energies for reference confs:" << std::endl;
         std::cout << std::fixed << std::setprecision(12);
         int nsub = space.size();
         for(int i=0; i<nsub; i++){
            std::cout << "i = " << i << " state = " << space[i]
               << " e = " << fock::get_Hii(space[i],int2e,int1e)+ecore 
               << std::endl;
         }
         
         // selected CISD space
         double eps1 = schd.ci.eps0;
         std::vector<double> cmax(nsub,1.0);
         expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.ci.flip);
         nsub = space.size();
         // set up initial states
         if(schd.ci.nroots > nsub) tools::exit("error: subspace is too small in sci::init_ciwf!");
         // diagonalize
         linalg::matrix<Tm> H = fock::get_Hmat(space, int2e, int1e, ecore);
         std::vector<double> esol(nsub);
         linalg::matrix<Tm> vsol;
         linalg::eig_solver(H, esol, vsol);
         // save
         int neig = schd.ci.nroots;
         es.resize(neig);
         vs.resize(nsub, neig);
         linalg::xcopy(neig, esol.data(), es.data());
         linalg::xcopy(nsub*neig, vsol.data(), vs.data());
         // print
         std::cout << std::fixed << std::setprecision(12);
         for(int i=0; i<neig; i++){
            std::cout << "i = " << i << " e = " << es[i] << std::endl; 
         }
         auto t1 = tools::get_time();
         tools::timing("sci::init_ciwf", t0, t1);
      }

   // selected CI procedure
   template <typename Tm>
      void ci_solver(const input::schedule& schd,
            fci::sparse_hamiltonian<Tm>& sparseH,
            std::vector<double>& es,
            linalg::matrix<Tm>& vs,
            fock::onspace& space,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore){
         const bool Htype = tools::is_complex<Tm>();
         auto t0 = tools::get_time();
         std::cout << "\nsci::ci_solver Htype=" << Htype << std::endl;

         // set up head-bath table
         heatbath_table<Tm> hbtab(int2e, int1e);
        
         // set up intial configurations
         std::vector<double> esol;
         linalg::matrix<Tm> vsol;
         std::unordered_set<fock::onstate> varSpace;
         init_ciwf(esol, vsol, space, varSpace, hbtab, schd, int2e, int1e, ecore);
        
         // set up auxilliary data structure   
         sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype);
        
         // start increment selected CI subspace
         bool ifconv = false;
         int nsub = space.size(); 
         int neig = schd.ci.nroots;
         for(int iter=0; iter<schd.ci.maxiter; iter++){
            std::cout << "\n---------------------" << std::endl;
            std::cout << "iter=" << iter << " eps1=" << std::scientific << schd.ci.eps1[iter] << std::endl;
            std::cout << "---------------------" << std::endl;
            double eps1 = schd.ci.eps1[iter];
            // compute cmax[i] = \sqrt{\sum_j|vj[i]|^2} for screening
            std::vector<double> cmax(nsub,0.0);
            for(int j=0; j<neig; j++){
               for(int i=0; i<nsub; i++){
                  cmax[i] += std::norm(vsol(i,j));
               }
            }
            std::transform(cmax.begin(), cmax.end(), cmax.begin(),
                  [](const double& x){ return std::pow(x,0.5); });
            // expand 
            expand_varSpace(space, varSpace, hbtab, cmax, eps1, schd.ci.flip);
            int nsub0 = nsub;
            nsub = space.size(); // nsub >= nsub0
            // update auxilliary data structure 
            sparseH.get_hamiltonian(space, int2e, int1e, ecore, Htype, nsub0);
            // set up Davidson solver 
            linalg::dvdsonSolver<Tm> solver(nsub, neig, schd.ci.crit_v, schd.ci.maxcycle);
            solver.Diag = sparseH.diag.data();
            using std::placeholders::_1;
            using std::placeholders::_2;
            solver.HVec = std::bind(&fci::get_Hx<Tm>, _1, _2, std::cref(sparseH));
            // copy previous initial guess
            linalg::matrix<Tm> v0(nsub, neig);
            for(int j=0; j<neig; j++){
               linalg::xcopy(nsub0, vsol.col(j), v0.col(j));
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
               conv[i] = std::abs(esol1[i]-esol[i]) < schd.ci.deltaE; 
               std::vector<Tm> vtmp(vsol1.col(i),vsol1.col(i)+nsub);
               double SvN = fock::coeff_entropy(vtmp); 
               std::cout << "sci: iter=" << iter
                  << " eps1=" << std::scientific << std::setprecision(3) << schd.ci.eps1[iter]
                  << " nsub=" << nsub 
                  << " i=" << i 
                  << " e=" << std::fixed << std::setprecision(12) << esol1[i] 
                  << " de=" << std::scientific << std::setprecision(3) << esol1[i]-esol[i] 
                  << " conv=" << conv[i] 
                  << " SvN=" << SvN
                  << std::endl;
               fock::coeff_analysis(vtmp);
               if(i != neig-1) std::cout << std::endl;
            }
            esol = esol1;
            vsol = vsol1;
            ifconv = (count(conv.begin(), conv.end(), true) == neig);
            if(iter>=schd.ci.miniter && ifconv){
               std::cout << "\nsci convergence is achieved for threshold deltaE=" 
                  << std::scientific << schd.ci.deltaE 
                  << std::endl;
               break;
            }
         } // iter

         // check final convergence
         if(!ifconv){
            std::cout << "\nsci convergence failure: out of maxiter=" << schd.ci.maxiter 
               << " for threshsold deltaE=" << std::scientific << schd.ci.deltaE
               << std::endl;
         }
         std::cout << std::endl;

         // finally save results
         es.resize(neig);
         vs.resize(nsub,neig);
         linalg::xcopy(neig, esol.data(), es.data());
         linalg::xcopy(nsub*neig, vsol.data(), vs.data());
         auto t1 = tools::get_time();
         tools::timing("sci::ci_solver", t0, t1);
      }

} // sci

#endif
