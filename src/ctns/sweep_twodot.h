#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/dvdson.h"
#include "../core/linalg.h"
#include "ctns_qdpt.h"
#include "sweep_dvdson.h"
//#include "sweep_onedot_ham.h"
//#include "sweep_onedot_guess.h"
//#include "sweep_decimation.h"

namespace ctns{

template <typename Km>
void sweep_twodot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore){
   using Tm = typename Km::dtype;
   const int isym = Km::isym;
   const bool ifkr = kind::is_kramers<Km>();
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();
   std::cout << "ctns::sweep_twodot" << std::endl;

   // 0. processing partition & symmetry
   auto dbond = sweeps.seq[ibond];
   auto p0 = dbond.p0;
   auto p1 = dbond.p1;
   auto p = dbond.p;
   auto cturn = dbond.cturn;
   std::vector<int> suppc1, suppc2, suppl, suppr;
   qbond qc1, qc2, ql, qr;
   if(!cturn){
      //
      //       |    |
      //    ---p0---p1---
      //
      suppc1 = icomb.get_suppc(p0, debug_sweep); 
      suppc2 = icomb.get_suppc(p1, debug_sweep); 
      suppl  = icomb.get_suppl(p0, debug_sweep);
      suppr  = icomb.get_suppr(p1, debug_sweep);
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
      suppc1 = icomb.get_suppc(p1, debug_sweep); 
      suppc2 = icomb.get_suppr(p1, debug_sweep); 
      suppl  = icomb.get_suppl(p0, debug_sweep);
      suppr  = icomb.get_suppr(p0, debug_sweep);
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
   qc1.print("qc1", debug_sweep);
   qc2.print("qc2", debug_sweep);
   ql.print("ql", debug_sweep);
   qr.print("qr", debug_sweep);

   // 1. load operators 
   oper_dict<Tm> c1qops, c2qops, lqops, rqops;
   if(!cturn){
      oper_load_qops(icomb, p0, schd.scratch, 'c', c1qops);
      oper_load_qops(icomb, p1, schd.scratch, 'c', c2qops);
      oper_load_qops(icomb, p0, schd.scratch, 'l', lqops );
      oper_load_qops(icomb, p1, schd.scratch, 'r', rqops );  
   }else{
      oper_load_qops(icomb, p1, schd.scratch, 'c', c1qops);
      oper_load_qops(icomb, p1, schd.scratch, 'r', c2qops);
      oper_load_qops(icomb, p0, schd.scratch, 'l', lqops );
      oper_load_qops(icomb, p0, schd.scratch, 'r', rqops );  
   }
   if(debug_sweep){
      const int level = 0;
      oper_display(c1qops, "c1qops", level);
      oper_display(c2qops, "c2qops", level);
      oper_display(lqops, "lqops", level);
      oper_display(rqops, "rqops", level);
   }
   timing.ta = tools::get_time();

   // 2. Davidson solver for wf
   qsym sym_state = (isym == 1)? qsym(schd.nelec) : qsym(schd.nelec, schd.twoms);
   qtensor4<Tm> wf(sym_state, qc1, qc2, ql, qr);
   if(debug_sweep) std::cout << "dim(localCI)=" << wf.get_dim() << std::endl;

/*
   // 2.1 Hdiag 
   int nsub = wf.get_dim();
   int neig = sweeps.nstates;
   std::vector<double> diag(nsub,1.0);
   diag = onedot_Hdiag(ifkr, cqops, lqops, rqops, ecore, wf);
   timing.tb = tools::get_time();
   
   // 2.2 Solve local problem: Hc=cE
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);
   using std::placeholders::_1;
   using std::placeholders::_2;
   auto HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, 
           	    std::cref(isym), std::cref(ifkr), 
           	    std::ref(cqops), std::ref(lqops), std::ref(rqops), 
           	    std::cref(int2e), std::cref(int1e), std::cref(ecore), 
           	    std::ref(wf));
   if(!ifkr){
      // without kramers restriction
      linalg::dvdsonSolver<Tm> solver(nsub, neig, sweeps.ctrls[isweep].eps, schd.maxcycle);
      solver.Diag = diag.data();
      solver.HVec = HVec;
      if(schd.cisolver == 0){
         solver.solve_diag(eopt.data(), vsol.data(), true); // full diagonalization for debug
      }else if(schd.cisolver == 1){ // davidson
         if(!sweeps.guess){
            solver.solve_iter(eopt.data(), vsol.data()); // davidson without initial guess
         }else{     
            // load initial guess from previous opt
            if(icomb.psi.size() == 0) onedot_guess_psi0(icomb,neig); // starting guess 
            assert(icomb.psi.size() == neig);
            assert(icomb.psi[0].get_dim() == nsub);
            std::vector<Tm> v0(nsub*neig);
            for(int i=0; i<neig; i++){
               icomb.psi[i].to_array(&v0[nsub*i]);
            }
            int nindp = linalg::get_ortho_basis(nsub, neig, v0); // reorthogonalization
            assert(nindp == neig);
            solver.solve_iter(eopt.data(), vsol.data(), v0.data());
         }
      }
   }else{
      // kramers restricted (currently works only for iterative with guess!) 
      assert(schd.cisolver == 1 && sweeps.guess);
      kr_dvdsonSolver<Tm,qtensor3<Tm>> solver(nsub, neig, sweeps.ctrls[isweep].eps, schd.maxcycle, 
		      			      (schd.nelec)%2, wf);
      solver.Diag = diag.data();
      solver.HVec = HVec;
      // load initial guess from previous opt
      if(icomb.psi.size() == 0) onedot_guess_psi0(icomb,neig); // starting guess 
      std::vector<Tm> v0;
      solver.init_guess(icomb.psi, v0);
      solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   } // ifkr
   timing.tc = tools::get_time();
*/
/*
   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = sweeps.nstates;
   auto diag = tns::get_twodot_Hdiag(c1qops, c2qops, lqops, rqops, int2e, ecore, wf);
   auto tb = tools::get_time();
   
   dvdsonSolver solver;
   solver.iprt = 1;
   solver.crit_v = ctrl.eps;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(tns::get_twodot_Hx, _1, _2, 
		      cref(icomb), cref(p),
		      ref(c1qops), ref(c2qops), ref(lqops), ref(rqops), 
		      cref(int2e), cref(int1e), cref(ecore), 
		      ref(wf));
   // solve
   eopt.resize(neig);
   matrix vsol(nsub,neig);
   // load initial guess from previous opt
   vector<double> v0(nsub*neig);
   if(icomb.psi.size() == 0) initial_onedot(icomb);
   assert(icomb.psi.size() == neig);
   initial_twodot(icomb, dbond, nsub, neig, wf, v0);
   int nindp = get_ortho_basis(nsub, neig, v0); // reorthogonalization
   assert(nindp == neig);
   //solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   auto tc = tools::get_time();

   // 3. decimation & renormalize operators
   twodot_decimation(sweeps, isweep, ibond, icomb, vsol, wf, 
		     cqops, lqops, rqops, int2e, int1e, schd.scratch);
*/

   timing.t1 = tools::get_time();
   std::cout << "timing for ctns::sweep_twodot : " << std::setprecision(2) 
             << tools::get_duration(timing.t1-timing.t0) << " s" << std::endl;
   timing.analysis();
}

} // ctns

#endif
