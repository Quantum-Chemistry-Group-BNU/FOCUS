#ifndef SWEEP_UTIL_H
#define SWEEP_UTIL_H

#include "../core/tools.h"
#include "../core/dvdson.h"
#include "../core/linalg.h"
#include "ctns_qdpt.h"
#include "sweep_ham_onedot.h"
#include "sweep_decimation.h"
#include "sweep_renorm.h"

namespace ctns{

using tm = std::chrono::high_resolution_clock::time_point;
void sweep_timing(const std::vector<tm> ts){
   double dt0 = tools::get_duration(ts[1]-ts[0]); // t(procs)
   double dt1 = tools::get_duration(ts[2]-ts[1]); // t(hdiag)
   double dt2 = tools::get_duration(ts[3]-ts[2]); // t(dvdsn)
   double dt3 = tools::get_duration(ts[4]-ts[3]); // t(decim)
   double dt4 = tools::get_duration(ts[5]-ts[4]); // t(renrm)
   double dt  = tools::get_duration(ts[5]-ts[0]); // total
   std::cout << " t(procs) = " << std::scientific << std::setprecision(2) << dt0 << " s "
	     << " per = " << std::defaultfloat << dt0/dt*100 << std::endl;
   std::cout << " t(hdiag) = " << std::scientific << std::setprecision(2) << dt1 << " s "
	     << " per = " << std::defaultfloat << dt1/dt*100 << std::endl;
   std::cout << " t(dvdsn) = " << std::scientific << std::setprecision(2) << dt2 << " s "
	     << " per = " << std::defaultfloat << dt2/dt*100 << std::endl;
   std::cout << " t(decim) = " << std::scientific << std::setprecision(2) << dt3 << " s "
	     << " per = " << std::defaultfloat << dt3/dt*100 << std::endl;
   std::cout << " t(renrm) = " << std::scientific << std::setprecision(2) << dt4 << " s "
	     << " per = " << std::defaultfloat << dt4/dt*100 << std::endl;
}

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
   auto t0 = tools::get_time();
   std::cout << "ctns::sweep_onedot" << std::endl;
   
   // 0. preprocessing 
   // processing partition & symmetry
   auto dbond = sweeps.seq[ibond];
   auto p = dbond.p;
   auto suppc = icomb.get_suppc(p); 
   auto suppl = icomb.get_suppl(p);
   auto suppr = icomb.get_suppr(p);
   int sc = suppc.size();
   int sl = suppl.size();
   int sr = suppr.size();
   // processing symmetry
   qbond qc, ql, qr;
   qc = icomb.get_qc(p); 
   ql = icomb.get_ql(p);
   qr = icomb.get_qr(p);
   qc.print("qc");
   ql.print("ql");
   qr.print("qr");
   // wavefunction to be computed
   qsym sym_state = (isym == 1)? qsym(schd.nelec) : qsym(schd.nelec, schd.twoms);
   qtensor3<Tm> wf(sym_state, qc, ql, qr, {1,1,1});
   std::cout << "dimCI=" << wf.get_dim() << std::endl;

   // 1. load operators 
   oper_dict<Tm> cqops, lqops, rqops;
   oper_load_qops(icomb, p, schd.scratch, 'c', cqops);
   oper_load_qops(icomb, p, schd.scratch, 'l', lqops);
   oper_load_qops(icomb, p, schd.scratch, 'r', rqops);
   oper_display(cqops, "cqops", 1);
   oper_display(lqops, "lqops", 1);
   oper_display(rqops, "rqops", 1);
   auto ta = tools::get_time();

   // 2. Davidson solver for wf
   // 2.1 Hdiag 
   int nsub = wf.get_dim();
   int neig = sweeps.nstate;
   std::vector<double> diag(nsub,1.0);
   diag = onedot_Hdiag(ifkr, cqops, lqops, rqops, ecore, wf);
 
   // debug 
   for(const auto& p : diag){
      std::cout << p << " ";
   }
   std::cout << std::endl;

   auto tb = tools::get_time();
   
   // 2.2 Solve Hc=cE
   linalg::dvdsonSolver<Tm> solver;
   solver.iprt = 1;
   solver.crit_v = sweeps.ctrls[isweep].eps;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(&ctns::onedot_Hx<Tm>, _1, _2, 
		      std::cref(isym), std::cref(ifkr), 
		      std::ref(cqops), std::ref(lqops), std::ref(rqops), 
		      std::cref(int2e), std::cref(int1e), std::cref(ecore), 
		      std::ref(wf));
   // solve local problem
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);
   if(schd.cisolver == 0){
      solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   }else if(schd.cisolver == 1){ // davidson
      // guess or not
 /*
   // load initial guess from previous opt
   vector<double> v0(nsub*neig);
   if(icomb.psi.size() == 0) initial_onedot(icomb); 
   assert(icomb.psi.size() == neig);
   assert(icomb.psi[0].get_dim() == nsub);
   for(int i=0; i<neig; i++){
      icomb.psi[i].to_array(&v0[nsub*i]);
   }
   int nindp = get_ortho_basis(nsub, neig, v0); // reorthogonalization
   assert(nindp == neig);
   //solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   solver.solve_iter(eopt.data(), vsol.data(), v0.data());
*/     
   }else{
      std::cout << "error: no such option for cisolver=" << schd.cisolver << std::endl;
   }
   auto tc = tools::get_time();

   // 3. decimation
   decimation_onedot(sweeps, isweep, ibond, icomb, vsol, wf, cqops, lqops, rqops);
   auto td = tools::get_time();

   // 4. renormalize operators
   renorm_onedot(sweeps.seq[ibond], icomb, cqops, lqops, rqops, int2e, int1e, schd.scratch);
   auto t1 = tools::get_time();
   std::cout << "timing for ctns::sweep_onedot : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
   
   sweep_timing({t0,ta,tb,tc,td,t1});
   
}


/*
template <typename Tm>
void opt_sweep_twodot(const input::schedule& schd,
		      const input::sweep_ctrl& ctrl,
                      comb<Tm>& icomb,
		      const directed_bond& dbond,
                      const integral::two_body<Tm>& int2e,
                      const integral::one_body<Tm>& int1e,
                      const double ecore,
		      dot_result& result){
   auto t0 = tools::get_time();
   cout << "tns::opt_twodot";

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
   int neig = schd.nroots;
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

void tns::opt_finaldot(const input::schedule& schd,
		       comb& icomb,
		       const integral::two_body& int2e,
	               const integral::one_body& int1e,
		       const double ecore){
   cout << "\ntns::opt_finaldot" << endl;
   // use one dot algorithm to produce a final wavefunction in 
   // right canonical form stored in icomb.rsites for later usage 
   auto p0 = make_pair(0,0);
   auto p1 = make_pair(1,0);
   auto dbond = make_tuple(p0,p1,false);
   input::sweep_ctrl ctrl = {0, 1, 4*schd.nroots, 1.e-5, 0.0};
   vector<double> eopt;
   double dwt;
   int deff;
   opt_onedot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, 
	      eopt, dwt, deff);
   // convert wfs on the last dot to rsite0
   icomb.rsites[p0] = get_rsite0(icomb.psi);
}

*/

} // ctns

#endif
