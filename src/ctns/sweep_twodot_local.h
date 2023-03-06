#ifndef SWEEP_TWODOT_LOCAL_H
#define SWEEP_TWODOT_LOCAL_H

#include "oper_functors.h"
#include "sweep_twodot_guess.h"

namespace ctns{

   // local CI solver	
   template <typename Km>
      void twodot_localCI(comb<Km>& icomb,
            const input::schedule& schd,
            const double eps,
            const int parity,
            const size_t ndim,
            const int neig,
            std::vector<double>& diag,
            HVec_type<typename Km:: dtype> HVec,
            std::vector<double>& eopt,
            linalg::matrix<typename Km::dtype>& vsol,
            int& nmvp,
            stensor4<typename Km::dtype>& wf,
            const directed_bond& dbond){
         using Tm = typename Km::dtype;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // without kramers restriction
         assert(Km::ifkr == false);
         pdvdsonSolver_nkr<Tm> solver(ndim, neig, eps, schd.ctns.maxcycle);
         solver.iprt = schd.ctns.verbose;
         solver.nbuff = schd.ctns.nbuff;
         solver.damping = schd.ctns.damping;
         solver.Diag = diag.data();
         solver.HVec = HVec;
#ifndef SERIAL
         solver.world = icomb.world;
#endif
         if(schd.ctns.cisolver == 0){

            // full diagonalization for debug
            solver.solve_diag(eopt.data(), vsol.data(), true);

         }else if(schd.ctns.cisolver == 1){ 

            // davidson
            if(schd.ctns.guess == 0){
               // davidson without initial guess
               solver.solve_iter(eopt.data(), vsol.data()); 
            }else if(schd.ctns.guess == 1){     
               //------------------------------------
               // prepare initial guess     
               //------------------------------------
               std::vector<Tm> v0;
               if(rank == 0){
                  assert(icomb.cpsi.size() == neig);
                  // specific to twodot 
                  twodot_guess(icomb, dbond, ndim, neig, wf, v0);
                  // reorthogonalization
                  int nindp = linalg::get_ortho_basis(ndim, neig, v0.data()); 
                  assert(nindp == neig);
               }
               //------------------------------------
               solver.solve_iter(eopt.data(), vsol.data(), v0.data());
            }else{
               std::cout << "error: no such option for guess=" << schd.ctns.guess << std::endl;
               exit(1);
            }

         }
         nmvp = solver.nmvp;
      }

   template <>
      inline void twodot_localCI(comb<qkind::cNK>& icomb,
            const input::schedule& schd,
            const double eps,
            const int parity,
            const size_t ndim,
            const int neig,
            std::vector<double>& diag,
            HVec_type<std::complex<double>> HVec,
            std::vector<double>& eopt,
            linalg::matrix<std::complex<double>>& vsol,
            int& nmvp,
            stensor4<std::complex<double>>& wf,
            const directed_bond& dbond){
         using Tm = std::complex<double>;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // kramers restricted (currently works only for iterative with guess!) 
         assert(schd.ctns.cisolver == 1 && schd.ctns.guess == 1);
         pdvdsonSolver_kr<Tm,stensor4<Tm>> solver(ndim, neig, eps, schd.ctns.maxcycle, parity, wf); 
         solver.iprt = schd.ctns.verbose;
         solver.nbuff = schd.ctns.nbuff;
         solver.damping = schd.ctns.damping;
         solver.Diag = diag.data();
         solver.HVec = HVec;
#ifndef SERIAL
         solver.world = icomb.world;
#endif
         //------------------------------------
         // prepare initial guess     
         //------------------------------------
         std::vector<Tm> v0;
         if(rank == 0){
            assert(icomb.cpsi.size() == neig);
            // specific to twodot 
            twodot_guess(icomb, dbond, ndim, neig, wf, v0);
            // load initial guess from previous opt
            std::vector<stensor4<Tm>> psi4(neig);
            for(int i=0; i<neig; i++){
               psi4[i].init(wf.info);
               // need to copy, as v0 will be modified in solver.init_guess
               psi4[i].from_array(&v0[ndim*i]); 
            }
            solver.init_guess(psi4, v0);
            psi4.clear();
         }
         //------------------------------------
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
         nmvp = solver.nmvp;
      }

} // ctns

#endif
