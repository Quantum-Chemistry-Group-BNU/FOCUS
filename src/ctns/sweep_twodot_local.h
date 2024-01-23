#ifndef SWEEP_TWODOT_LOCAL_H
#define SWEEP_TWODOT_LOCAL_H

#include "oper_functors.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void twodot_guess_v0(comb<Qm,Tm>& icomb, 
            const directed_bond& dbond,
            const size_t ndim,
            const int neig,
            stensor4<Tm>& wf,
            std::vector<Tm>& v0){
         const bool debug = true;
         if(debug) std::cout << "ctns::twodot_guess ";
         auto pdx0 = icomb.topo.rindex.at(dbond.p0);
         auto pdx1 = icomb.topo.rindex.at(dbond.p1);
         assert(icomb.cpsi.size() == neig);
         v0.resize(ndim*neig);
         if(dbond.forward){
            if(!dbond.is_cturn()){

               if(debug) std::cout << "|lc1>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[l,a,c1] => cwf[lc1,a]
                  auto cwf = icomb.cpsi[i].merge_lc(); 
                  // cwf[lc1,a]*r[a,r,c2] => wf3[lc1,r,c2]
                  auto wf3 = contract_qt3_qt2("l",icomb.sites[pdx1],cwf); 
                  // wf3[lc1,r,c2] => wf4[l,r,c1,c2]
                  auto wf4 = wf3.split_lc1(wf.info.qrow, wf.info.qmid);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               //
               //     c2
               //      |
               // c1---p1 
               //      |
               //  l---p0---r
               //     [psi]
               //
               if(debug) std::cout << "|lr>(comb)" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[l,r,a] => cwf[lr,a]		 
                  auto cwf = icomb.cpsi[i].merge_lr(); // on backone
                                                       // r[a,c2,c1] => r[a,c1c2], cwf[lr,a]*r[a,c1c2] => wf2[lr,c1c2]
                  auto wf2 = cwf.dot(icomb.sites[pdx1].merge_cr());
                  // wf2[lr,c1c2] => wf4[l,r,c1,c2] 
                  auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            } // cturn
         }else{
            if(!dbond.is_cturn()){

               if(debug) std::cout << "|c2r>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[a,r,c2] => cwf[a,c2r]
                  auto cwf = icomb.cpsi[i].merge_cr();
                  // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
                  auto wf3 = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.P());
                  // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
                  auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               //
               //     c2
               //      |
               // c1---p0 [psi]
               //      |
               //  l---p1---r
               //
               if(debug) std::cout << "|c1c2>(comb)" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[a,c2,c1] => cwf[a,c1c2]
                  auto cwf = icomb.cpsi[i].merge_cr(); // on branch
                                                       // l[l,r,a] => l[lr,a], l[lr,a]*cwf[a,c1c2] => wf2[lr,c1c2]
                  auto wf2 = icomb.sites[pdx0].merge_lr().dot(cwf);
                  // wf2[lr,c1c2] => wf4[l,r,c1,c2]
                  auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
                  wf4.permCR_signed(); // back to backbone
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            } // cturn
         } // forward
      }

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
            stensor4<Tm>& wf,
            const directed_bond& dbond,
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
                  assert(icomb.cpsi.size() == neig);
                  // specific to twodot 
                  twodot_guess_v0(icomb, dbond, ndim, neig, wf, v0);
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
            stensor4<std::complex<double>>& wf,
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
