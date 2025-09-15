#ifndef SWEEP_TWODOT_LOCAL_H
#define SWEEP_TWODOT_LOCAL_H

#include "oper_functors.h"
#include "sweep_twodot_guess.h"

namespace ctns{

   // local CI solver	
   template <typename Qm, typename Tm>
      void twodot_localCI(comb<Qm,Tm>& icomb,
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
            qtensor4<Qm::ifabelian,Tm>& wf,
            const directed_bond& dbond,
            dot_timing& timing){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif

         // without kramers restriction
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
            solver.solve_diag(eopt.data(), vsol.data(), schd.ctns.debug_hmat, true);

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
                  assert(icomb.cpsi.size() == neig);
                  // specific to twodot 
                  twodot_guess_v0(icomb, dbond, ndim, neig, wf, v0);
                  // reorthogonalization
                  int nindp = linalg::get_ortho_basis(ndim, neig, v0.data()); 
                  if(nindp != neig){
                     std::cout << "error: nindp=" << nindp << " does not match neig=" << neig << std::endl;
                     exit(1);
                  } 
               }
               //------------------------------------
               auto t1 = tools::get_time();
               timing.dtb5 = tools::get_duration(t1-t0);
               solver.solve_iter(eopt.data(), vsol.data(), v0.data());
            }else{
               std::cout << "error: no such option for guess=" << schd.ctns.guess << std::endl;
               exit(1);
            }

         }
         nmvp = solver.nmvp;
         timing.dtb6 = solver.t_cal - oper_timer.tcpugpu - oper_timer.tcommgpu; 
         timing.dtb7 = oper_timer.tcpugpu;
         timing.dtb8 = oper_timer.tcommgpu;
         timing.dtb9 = solver.t_comm;
         timing.dtb10 = solver.t_rest;
      }

   template <>
      inline void twodot_localCI(comb<qkind::qNK,std::complex<double>>& icomb,
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
            qtensor4<true,std::complex<double>>& wf,
            const directed_bond& dbond,
            dot_timing& timing){
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
            assert(icomb.cpsi.size() == neig);
            // specific to twodot 
            twodot_guess_v0(icomb, dbond, ndim, neig, wf, v0);
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
         auto t1 = tools::get_time();
         timing.dtb5 = tools::get_duration(t1-t0);
         // solve
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
         nmvp = solver.nmvp;
         timing.dtb6 = solver.t_cal - oper_timer.tcpugpu - oper_timer.tcommgpu; 
         timing.dtb7 = oper_timer.tcpugpu;
         timing.dtb8 = oper_timer.tcommgpu;
         timing.dtb9 = solver.t_comm;
         timing.dtb10 = solver.t_rest;
      }

} // ctns

#endif
