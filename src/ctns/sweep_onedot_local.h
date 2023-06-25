#ifndef SWEEP_ONEDOT_LOCAL_H
#define SWEEP_ONEDOT_LOCAL_H

#include "oper_functors.h"

namespace ctns{

   // local CI solver	
   template <typename Qm, typename Tm>
      void onedot_localCI(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const double eps,
            const int parity,
            const size_t ndim,
            const int neig,
            const double* diag,
            HVec_type<Tm> HVec,
            std::vector<double>& eopt,
            linalg::matrix<Tm>& vsol,
            int& nmvp,
            stensor3<Tm>& wf,
            dot_timing& timing){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // without kramers restriction
         assert(Qm::ifkr == false);
         pdvdsonSolver_nkr<Tm> solver(ndim, neig, eps, schd.ctns.maxcycle);
         solver.iprt = schd.ctns.verbose;
         solver.nbuff = schd.ctns.nbuff;
         solver.damping = schd.ctns.damping;
         solver.precond = schd.ctns.precond;
         solver.ifnccl = schd.ctns.ifnccl;
         solver.Diag = const_cast<double*>(diag);
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
               auto t0 = tools::get_time();
               std::vector<Tm> v0;
               if(rank == 0){ 
                  assert(icomb.cpsi.size() == neig && icomb.cpsi[0].size() == ndim);
                  // load initial guess from previous opt
                  v0.resize(ndim*neig);
                  for(int i=0; i<neig; i++){
                     icomb.cpsi[i].to_array(&v0[ndim*i]);
                  }
                  // reorthogonalization
                  int nindp = linalg::get_ortho_basis(ndim, neig, v0.data()); 
                  assert(nindp == neig);
               }
               //------------------------------------
               auto t1 = tools::get_time();
               timing.dtb7 = tools::get_duration(t1-t0);
               solver.solve_iter(eopt.data(), vsol.data(), v0.data());
            }else{
               std::cout << "error: no such option for guess=" << schd.ctns.guess << std::endl;
               exit(1);
            }

         }
         nmvp = solver.nmvp;
         timing.dtb8 = solver.t_cal; 
         timing.dtb9 = solver.t_comm;
         timing.dtb10 = solver.t_rest;
      }

   template <>
      inline void onedot_localCI(comb<qkind::qNK,std::complex<double>>& icomb,
            const input::schedule& schd,
            const double eps,
            const int parity,
            const size_t ndim,
            const int neig,
            const double* diag,
            HVec_type<std::complex<double>> HVec,
            std::vector<double>& eopt,
            linalg::matrix<std::complex<double>>& vsol,
            int& nmvp,
            stensor3<std::complex<double>>& wf,
            dot_timing& timing){
         using Tm = std::complex<double>;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // kramers restricted (currently works only for iterative with guess!)
         assert(schd.ctns.cisolver == 1 && schd.ctns.guess == 1);
         pdvdsonSolver_kr<Tm,stensor3<Tm>> solver(ndim, neig, eps, schd.ctns.maxcycle, parity, wf); 
         solver.iprt = schd.ctns.verbose;
         solver.nbuff = schd.ctns.nbuff;
         solver.damping = schd.ctns.damping;
         solver.precond = schd.ctns.precond;
         solver.ifnccl = schd.ctns.ifnccl;
         solver.Diag = const_cast<double*>(diag);
         solver.HVec = HVec;
#ifndef SERIAL
         solver.world = icomb.world;
#endif
         //------------------------------------
         // prepare initial guess     
         //------------------------------------
         auto t0 = tools::get_time();
         std::vector<Tm> v0;
         if(rank == 0){
            assert(icomb.cpsi.size() == neig && icomb.cpsi[0].size() == ndim);
            // load initial guess from previous opt
            solver.init_guess(icomb.cpsi, v0);
         }
         //------------------------------------
         auto t1 = tools::get_time();
         timing.dtb7 = tools::get_duration(t1-t0);
         // solve
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
         nmvp = solver.nmvp;
         timing.dtb8 = solver.t_cal; 
         timing.dtb9 = solver.t_comm;
         timing.dtb10 = solver.t_rest;
      }

} // ctns

#endif
