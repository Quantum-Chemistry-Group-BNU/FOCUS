#ifndef SWEEP_UTIL_H
#define SWEEP_UTIL_H

#include "../core/tools.h"
#include "../core/dvdson.h"
#include "../core/linalg.h"
#include "ctns_qdpt.h"
#include "sweep_ham_onedot.h"
#include "sweep_decimation.h"
#include "sweep_guess.h"
#include "sweep_dvdson.h"

namespace ctns{

const bool debug_sweep = false;
extern const bool debug_sweep;

template <typename Km>
void sweep_onedot(const input::schedule& schd,
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
   std::cout << "ctns::sweep_onedot" << std::endl;
   
   // 0. preprocessing 
   // processing partition & symmetry
   auto dbond = sweeps.seq[ibond];
   auto p = dbond.p;
   auto suppc = icomb.get_suppc(p, debug_sweep); 
   auto suppl = icomb.get_suppl(p, debug_sweep);
   auto suppr = icomb.get_suppr(p, debug_sweep);
   int sc = suppc.size();
   int sl = suppl.size();
   int sr = suppr.size();
   // processing symmetry
   qbond qc, ql, qr;
   qc = icomb.get_qc(p); 
   ql = icomb.get_ql(p);
   qr = icomb.get_qr(p);
   qc.print("qc", debug_sweep);
   ql.print("ql", debug_sweep);
   qr.print("qr", debug_sweep);
   // wavefunction to be computed
   qsym sym_state = (isym == 1)? qsym(schd.nelec) : qsym(schd.nelec, schd.twoms);
   qtensor3<Tm> wf(sym_state, qc, ql, qr, {1,1,1});
   if(debug_sweep) std::cout << "dim(localCI)=" << wf.get_dim() << std::endl;

   // 1. load operators 
   oper_dict<Tm> cqops, lqops, rqops;
   oper_load_qops(icomb, p, schd.scratch, 'c', cqops);
   oper_load_qops(icomb, p, schd.scratch, 'l', lqops);
   oper_load_qops(icomb, p, schd.scratch, 'r', rqops);
   if(debug_sweep){
      const int level = 0;
      oper_display(cqops, "cqops", level);
      oper_display(lqops, "lqops", level);
      oper_display(rqops, "rqops", level);
   }
   timing.ta = tools::get_time();

   // 2. Davidson solver for wf
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
            if(icomb.psi.size() == 0) guess_onedot_psi0(icomb,neig); // starting guess 
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
      if(icomb.psi.size() == 0) guess_onedot_psi0(icomb,neig); // starting guess 
      std::vector<Tm> v0;
      solver.init_guess(icomb.psi, v0);
      solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   } // ifkr
   timing.tc = tools::get_time();

   // 3. decimation & renormalize operators
   decimation_onedot(sweeps, isweep, ibond, icomb, vsol, wf, 
		     cqops, lqops, rqops, int2e, int1e, schd.scratch);

   timing.t1 = tools::get_time();
   std::cout << "timing for ctns::sweep_onedot : " << std::setprecision(2) 
             << tools::get_duration(timing.t1-timing.t0) << " s" << std::endl;
   timing.analysis();
}

// use one dot algorithm to produce a final wavefunction
// in right canonical form for later usage 
template <typename Km>
void sweep_rwfuns(const input::schedule& schd,
		  comb<Km>& icomb,
		  const integral::two_body<typename Km::dtype>& int2e,
	          const integral::one_body<typename Km::dtype>& int1e,
		  const double ecore){
   std::cout << "ctns::sweep_rwfuns" << std::endl;

   // perform an additional onedot opt  
   const int dcut1 = -1;
   const double eps = schd.combsweep[schd.maxsweep-1].eps; // take the last eps 
   const double noise = 0.0; 
   input::sweep_ctrl ctrl = {0, 1, dcut1, eps, noise};
   auto p0 = std::make_pair(0,0);
   auto p1 = std::make_pair(1,0);
   auto cturn = icomb.topo.is_cturn(p0,p1);
   auto dbond = directed_bond(p0,p1,0,p1,cturn);
   sweep_data sweeps({dbond}, schd.nstates, schd.guess, 0, 1, {ctrl});
   sweep_onedot(schd, sweeps, 0, 0, icomb, int2e, int1e, ecore);

   std::cout << "deal with site0 by decimation for rsite0 & rwfuns" << std::endl;
   auto wf = icomb.psi[0];
   auto qprod = qmerge(wf.qmid, wf.qcol);
   auto qcr = qprod.first;
   auto dpt = qprod.second;
   // build RDM 
   qtensor2<typename Km::dtype> rdm(qsym(), qcr, qcr);
   for(int i=0; i<schd.nstates; i++){
      rdm += icomb.psi[i].merge_cr().get_rdm_col();
   }
   // decimation
   const int dcut = schd.nstates;
   double dwt; 
   int deff;
   const bool ifkr = tools::is_complex<Km>();
   auto qt2 = decimation_row(rdm, dcut, dwt, deff, ifkr, wf.qmid, wf.qcol, dpt).T(); 
   icomb.rsites[p0] = qt2.split_cr(wf.qmid, wf.qcol, dpt);
   // form rwfuns
   auto& sym_state = icomb.psi[0].sym;
   qbond qrow({{sym_state, schd.nstates}});
   auto& qcol = qt2.qrow; 
   qtensor2<typename Km::dtype> rwfuns(qsym(), qrow, qcol, {0, 1});
   assert(qcol.size() == 1);
   int rdim = qrow.get_dim(0);
   int cdim = qcol.get_dim(0);
   for(int i=0; i<schd.nstates; i++){
      auto cwf = icomb.psi[i].merge_cr().dot(qt2.H()); // <-W[1,alpha]->
      for(int c=0; c<cdim; c++){
         rwfuns(0,0)(i,c) = cwf(0,0)(0,c);	      
      }
   }
   icomb.rwfuns = std::move(rwfuns);
}

/*
template <typename Tm>
void sweep_twodot(const input::schedule& schd,
		      const input::sweep_ctrl& ctrl,
                      comb<Tm>& icomb,
		      const directed_bond& dbond,
                      const integral::two_body<Tm>& int2e,
                      const integral::one_body<Tm>& int1e,
                      const double ecore,
		      dot_result& result){
   auto t0 = tools::get_time();
   cout << "ctns::opt_twodot";

   // 0. determine bond and site to be updated
   auto p0 = std::get<0>(dbond);
   auto p1 = std::get<1>(dbond);
   auto forward = std::get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   
   // 1. process symmetry information & operators for {|lmvr>}
   qbond qc1, qc2, ql, qr;
   oper_dict c1qops, c2qops, lqops, rqops;
   if(!cturn){
      qc1 = icomb.get_qc(p0);
      qc2 = icomb.get_qc(p1);
      ql  = icomb.get_ql(p0);
      qr  = icomb.get_qr(p1);
      c1qops = oper_get_cqops(icomb, p0, schd.scratch);
      c2qops = oper_get_cqops(icomb, p1, schd.scratch);
      lqops  = oper_get_lqops(icomb, p0, schd.scratch);
      rqops  = oper_get_rqops(icomb, p1, schd.scratch);  
   }else{
      qc1 = icomb.get_qc(p1);
      qc2 = icomb.get_qr(p1);
      ql  = icomb.get_ql(p0);
      qr  = icomb.get_qr(p0);
      c1qops = oper_get_cqops(icomb, p1, schd.scratch);
      c2qops = oper_get_rqops(icomb, p1, schd.scratch);
      lqops  = oper_get_lqops(icomb, p0, schd.scratch);
      rqops  = oper_get_rqops(icomb, p0, schd.scratch);  
   }
   string info = " c1:"+oper_dict_opnames(c1qops)+  
	         " c2:"+oper_dict_opnames(c2qops)+
	         " l:" +oper_dict_opnames(lqops)+
	         " r:" +oper_dict_opnames(rqops);

   // wavefunction to be computed 
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor4 wf(sym_state,qc1,qc2,ql,qr);
   cout << " dim=" << wf.get_dim() << info << endl;
   auto ta = tools::get_time();

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
   initial_twodot(icomb, dbond, wf, nsub, neig, v0);
   int nindp = get_ortho_basis(nsub, neig, v0); // reorthogonalization
   assert(nindp == neig);
   //solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
   auto tc = tools::get_time();
   
   // 3. decimation & renormalize operators
   decimation_twodot(icomb, dbond, ctrl.dcut, vsol, wf, dwt, deff);
   auto td = tools::get_time();
   
   oper_renorm_twodot(icomb, dbond, c1qops, c2qops, lqops, rqops, int2e, int1e, schd.scratch);
   auto t1 = tools::get_time();

   cout << "timing for tns::opt_twodot : " << setprecision(2) 
        << tools::get_duration(t1-t0) << " s" << endl;
   opt_timing_analysis({t0,ta,tb,tc,td,t1});
}

*/

} // ctns

#endif
