#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_guess.h"
#include "sweep_twodot_ham.h"
#include "sweep_twodot_renorm.h"

namespace ctns{

template <typename Km>
void twodot_localCI(comb<Km>& icomb,
		    const int nsub,
		    const int neig,
   		    std::vector<double>& diag,
		    HVec_type<typename Km:: dtype> HVec,
   		    std::vector<double>& eopt,
   		    linalg::matrix<typename Km::dtype>& vsol,
		    int& nmvp,
		    const int cisolver,
		    const bool guess,
		    const double eps,
		    const int maxcycle,
		    const int parity,
		    const directed_bond& dbond,
		    qtensor4<typename Km::dtype>& wf){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = typename Km::dtype;
   // without kramers restriction
   dvdsonSolver_nkr<Tm> solver(nsub, neig, eps, maxcycle);
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   if(cisolver == 0){
      solver.solve_diag(eopt.data(), vsol.data(), true); // full diagonalization for debug
   }else if(cisolver == 1){ // davidson
      if(!guess){
         solver.solve_iter(eopt.data(), vsol.data()); // davidson without initial guess
      }else{     
         std::vector<Tm> v0(nsub*neig);
	 if(rank == 0){
            if(icomb.psi.size() == 0) onedot_guess_psi0(icomb, neig); // starting guess 
            auto psi4 = twodot_guess(icomb, dbond, nsub, neig, wf);
            for(int i=0; i<neig; i++){
               // load initial guess from previous opt
               psi4[i].to_array(&v0[nsub*i]);
            }
            int nindp = linalg::get_ortho_basis(nsub, neig, v0); // reorthogonalization
            assert(nindp == neig);
	 }
         solver.solve_iter(eopt.data(), vsol.data(), v0.data());
      }
   }
   nmvp = solver.nmvp;
}
template <>
inline void twodot_localCI(comb<kind::cNK>& icomb,
		    const int nsub,
		    const int neig,
   		    std::vector<double>& diag,
		    HVec_type<std::complex<double>> HVec,
   		    std::vector<double>& eopt,
   		    linalg::matrix<std::complex<double>>& vsol,
		    int& nmvp,
		    const int cisolver,
		    const bool guess,
		    const double eps,
		    const int maxcycle,
		    const int parity,
		    const directed_bond& dbond,
		    qtensor4<std::complex<double>>& wf){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif
   using Tm = std::complex<double>;
   // kramers restricted (currently works only for iterative with guess!) 
   assert(cisolver == 1 && guess);
   dvdsonSolver_kr<Tm,qtensor4<Tm>> solver(nsub, neig, eps, maxcycle, parity, wf); 
   solver.Diag = diag.data();
   solver.HVec = HVec;
#ifndef SERIAL
   solver.world = icomb.world;
#endif
   std::vector<Tm> v0;
   if(rank == 0){
      if(icomb.psi.size() == 0) onedot_guess_psi0(icomb,neig); // starting guess 
      auto psi4 = twodot_guess(icomb, dbond, nsub, neig, wf);
      // load initial guess from previous opt
      solver.init_guess(psi4, v0);
   }
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   nmvp = solver.nmvp;
}

template <typename Km>
void sweep_twodot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::sweep_twodot" << std::endl;
   const int isym = Km::isym;
   const bool ifkr = kind::is_kramers<Km>();
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. processing partition & symmetry
   auto dbond = sweeps.seq[ibond];
   auto p0 = dbond.p0;
   auto p1 = dbond.p1;
   auto p = dbond.p;
   auto cturn = dbond.cturn;
   std::vector<int> suppc1, suppc2, suppl, suppr;
   qbond qc1, qc2, ql, qr;
   if(rank == 0 && debug_sweep) std::cout << "support info:" << std::endl;
   if(!cturn){
      //
      //       |    |
      //    ---p0---p1---
      //
      suppc1 = icomb.get_suppc(p0, rank == 0 && debug_sweep); 
      suppc2 = icomb.get_suppc(p1, rank == 0 && debug_sweep); 
      suppl  = icomb.get_suppl(p0, rank == 0 && debug_sweep);
      suppr  = icomb.get_suppr(p1, rank == 0 && debug_sweep);
      qc1 = icomb.get_qc(p0);
      qc2 = icomb.get_qc(p1);
      ql  = icomb.get_ql(p0);
      qr  = icomb.get_qr(p1);
   }else{
      //       |
      //    ---p1
      //       |
      //    ---p0---
      //
      suppc1 = icomb.get_suppc(p1, rank == 0 && debug_sweep); 
      suppc2 = icomb.get_suppr(p1, rank == 0 && debug_sweep); 
      suppl  = icomb.get_suppl(p0, rank == 0 && debug_sweep);
      suppr  = icomb.get_suppr(p0, rank == 0 && debug_sweep);
      qc1 = icomb.get_qc(p1);
      qc2 = icomb.get_qr(p1);
      ql  = icomb.get_ql(p0);
      qr  = icomb.get_qr(p0);
   }
   int sc1 = suppc1.size();
   int sc2 = suppc2.size();
   int sl = suppl.size();
   int sr = suppr.size();
   assert(sc1+sc2+sl+sr == icomb.topo.nphysical);
   if(rank == 0){
      if(debug_sweep) std::cout << "qbond info:" << std::endl;
      qc1.print("qc1", debug_sweep);
      qc2.print("qc2", debug_sweep);
      ql.print("ql", debug_sweep);
      qr.print("qr", debug_sweep);
   }

   // 1. load operators 
   using Tm = typename Km::dtype;
   oper_dict<Tm> c1qops, c2qops, lqops, rqops;
   if(!cturn){
      oper_load_qops(icomb, p0, schd.scratch, "c", c1qops);
      oper_load_qops(icomb, p1, schd.scratch, "c", c2qops);
      oper_load_qops(icomb, p0, schd.scratch, "l", lqops );
      oper_load_qops(icomb, p1, schd.scratch, "r", rqops );  
   }else{
      oper_load_qops(icomb, p1, schd.scratch, "c", c1qops);
      oper_load_qops(icomb, p1, schd.scratch, "r", c2qops);
      oper_load_qops(icomb, p0, schd.scratch, "l", lqops );
      oper_load_qops(icomb, p0, schd.scratch, "r", rqops );  
   }
   if(debug_sweep){
      for(int iproc=0; iproc<size; iproc++){
          if(rank == iproc){
             std::cout << "qops info: rank=" << rank << std::endl;
             const int level = 0;
             c1qops.print("c1qops", level);
             c2qops.print("c2qops", level);
             lqops.print("lqops", level);
             rqops.print("rqops", level);
	  }
#ifndef SERIAL
	  icomb.world.barrier();
#endif
      } // iproc
   }
   timing.ta = tools::get_time();

   // 2. Davidson solver for wf
   qsym sym_state = (isym == 1)? qsym(schd.nelec) : qsym(schd.nelec, schd.twoms);
   qtensor4<Tm> wf(sym_state, qc1, qc2, ql, qr);
   if(rank == 0 && debug_sweep) std::cout << "dim(localCI)=" << wf.get_dim() << std::endl;
   int nsub = wf.get_dim();
   int neig = sweeps.nstates;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);
   
   // 2.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   diag = twodot_Hdiag(ifkr, c1qops, c2qops, lqops, rqops, ecore, wf, size, rank);
#ifndef SERIAL
   // reduction of partial Hdiag: no need to broadcast, if only rank=0 
   // executes the preconditioning in Davidson's algorithm
   if(size > 1){
      std::vector<double> diag2(nsub);
      boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
      diag = diag2;
   }
#endif 
   timing.tb = tools::get_time();

   // 2.2 Solve local problem: Hc=cE
   using std::placeholders::_1;
   using std::placeholders::_2;
   auto HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, 
           	    std::cref(isym), std::cref(ifkr),
           	    std::ref(c1qops), std::ref(c2qops), std::ref(lqops), std::ref(rqops), 
           	    std::cref(int2e), std::cref(int1e), std::cref(ecore), 
           	    std::ref(wf), std::cref(size), std::cref(rank));
   twodot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, dbond, wf);
   timing.tc = tools::get_time();
   if(rank == 0) sweeps.print_eopt(isweep, ibond);

   // 3. decimation & renormalize operators
   twodot_renorm(sweeps, isweep, ibond, icomb, vsol, wf, 
		 c1qops, c2qops, lqops, rqops, int2e, int1e, schd.scratch);

   timing.t1 = tools::get_time();
   if(rank == 0){
      std::cout << "timing for ctns::sweep_twodot : " << std::setprecision(2) 
                << tools::get_duration(timing.t1-timing.t0) << " s" 
		<< " size=" << size << std::endl;
      timing.analysis();
   }
}

} // ctns

#endif
