#ifndef CTNS_OPT_UTIL_H
#define CTNS_OPT_UTIL_H

namespace ctns{
   
using sweep_result = std::pair<double,std::vector<double>>; // (dwt,eopt)

/*
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

void tns::opt_onedot(const input::schedule& schd,
		     const input::sweep_ctrl& ctrl,
                     comb& icomb,
		     const directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore,
		     vector<double>& eopt,
		     double& dwt,
		     int& deff){
   auto t0 = tools::get_time();
   cout << "tns::opt_onedot";

   // 0. determine bond and site to be updated
   auto p0 = std::get<0>(dbond);
   auto p1 = std::get<1>(dbond);
   auto forward = std::get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   
   // 1. process symmetry information & operators for {|lcr>}
   qsym_space qc, ql, qr;
   oper_dict cqops, lqops, rqops;
   qc = icomb.get_qc(p); 
   ql = icomb.get_ql(p);
   qr = icomb.get_qr(p);
   cqops = oper_get_cqops(icomb, p, schd.scratch);
   lqops = oper_get_lqops(icomb, p, schd.scratch);
   rqops = oper_get_rqops(icomb, p, schd.scratch);
   string info = " c:"+oper_dict_opnames(cqops)+   
	         " l:"+oper_dict_opnames(lqops)+
	         " r:"+oper_dict_opnames(rqops);
  
   // wavefunction to be computed
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   vector<bool> dir = {1,1,1};
   qtensor3 wf(sym_state,qc,ql,qr,dir);
   cout << " dim=" << wf.get_dim() << info << endl;
   auto ta = tools::get_time();

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = schd.nroots;
   auto diag = tns::get_onedot_Hdiag(cqops, lqops, rqops, int2e, ecore, wf);
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
   solver.HVec = bind(tns::get_onedot_Hx, _1, _2, 
		      cref(icomb), cref(p),
		      ref(cqops), ref(lqops), ref(rqops), 
		      cref(int2e), cref(int1e), cref(ecore), 
		      ref(wf));
   
   // solve local problem
   eopt.resize(neig);
   matrix vsol(nsub,neig);
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
   auto tc = tools::get_time();
   
   // 3. decimation & renormalize operators
   decimation_onedot(icomb, dbond, ctrl.dcut, vsol, wf, dwt, deff,
		     ctrl.noise, cqops, lqops, rqops);
   auto td = tools::get_time();

   oper_renorm_onedot(icomb, dbond, cqops, lqops, rqops, int2e, int1e, schd.scratch);
   
   auto t1 = tools::get_time();
   cout << "timing for tns::opt_onedot : " << setprecision(2) 
        << tools::get_duration(t1-t0) << " s" << endl;
   opt_timing_analysis({t0,ta,tb,tc,td,t1});
}

void tns::opt_twodot(const input::schedule& schd,
		     const input::sweep_ctrl& ctrl,
                     comb& icomb,
		     const directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore,
		     vector<double>& eopt,
		     double& dwt,
		     int& deff){
   auto t0 = tools::get_time();
   cout << "tns::opt_twodot";

   // 0. determine bond and site to be updated
   auto p0 = std::get<0>(dbond);
   auto p1 = std::get<1>(dbond);
   auto forward = std::get<2>(dbond);
   auto p = forward? p0 : p1;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   
   // 1. process symmetry information & operators for {|lmvr>}
   qsym_space qc1, qc2, ql, qr;
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

void tns::opt_timing_analysis(const vector<tns::tm> ts){
   double dt0 = tools::get_duration(ts[1]-ts[0]);
   double dt1 = tools::get_duration(ts[2]-ts[1]);
   double dt2 = tools::get_duration(ts[3]-ts[2]);
   double dt3 = tools::get_duration(ts[4]-ts[3]);
   double dt4 = tools::get_duration(ts[5]-ts[4]);
   double dt  = tools::get_duration(ts[5]-ts[0]); // total
   cout << "  t(procs)=" << scientific << setprecision(2) << dt0 << " s"
	<< "  per=" << defaultfloat << dt0/dt*100 << endl;
   cout << "  t(hdiag)=" << scientific << setprecision(2) << dt1 << " s"
	<< "  per=" << defaultfloat << dt1/dt*100<< endl;
   cout << "  t(dvdsn)=" << scientific << setprecision(2) << dt2 << " s"
	<< "  per=" << defaultfloat << dt2/dt*100<< endl;
   cout << "  t(decim)=" << scientific << setprecision(2) << dt3 << " s"
	<< "  per=" << defaultfloat << dt3/dt*100<< endl;
   cout << "  t(renrm)=" << scientific << setprecision(2) << dt4 << " s"
	<< "  per=" << defaultfloat << dt4/dt*100<< endl;
}
*/

void opt_sweep_summary(const input::schedule& schd,
		       const std::vector<directed_bond>& sweeps,
		       // current sweep
		       const std::vector<std::vector<double>>& eopt,
		       const std::vector<double>& dwt,
		       const std::vector<int>& deff,
		       // updated data
		       const int isweep,
		       const std::vector<double>& timing,
		       std::vector<sweep_result>& sweep_data){
   //
   // analysi of the current sweep (eopt,dwt,deff) and timing
   //
   std::cout << "\n" << tools::line_separator << std::endl;
   auto& ctrl = schd.combsweep[isweep];
   input::combsweep_print(ctrl);
   int size = sweeps.size();
   std::vector<double> emean(size,0.0);
   for(int i=0; i<size; i++){
      auto dbond = sweeps[i];
      auto p0 = std::get<0>(dbond);
      auto p1 = std::get<1>(dbond);
      auto forward = std::get<2>(dbond);
      auto p = forward? p0 : p1;
      std::cout << "i=" << i << " bond=" << p0 << "-" << p1 
           << " fw=" << forward
           << " deff=" << deff[i]
           << " dwt=" << std::showpos << std::scientific << std::setprecision(2) 
	   << dwt[i] << std::noshowpos;
      // print energy
      std::cout << std::defaultfloat << std::setprecision(12);
      int nstate = eopt[i].size();
      for(int j=0; j<nstate; j++){ 
         std::cout << " e" << j << ":" << eopt[i][j];
         emean[i] += eopt[i][j];
      }
      emean[i] /= nstate;
      std::cout << std::endl;
   }
   auto pos = std::min_element(emean.begin(), emean.end());
   auto minpos = std::distance(emean.begin(), pos);
   sweep_data[isweep].first = dwt[minpos];
   sweep_data[isweep].second = eopt[minpos];
   std::cout << "min energies at pos=" << minpos << std::endl;
   std::cout << "timing for sweep: " << std::setprecision(2) << timing[isweep] << " s" << std::endl; 
   std::cout << tools::line_separator << std::endl;
   //
   // print all previous optimized results - sweep_data
   //
   std::cout << "\n" << tools::line_separator2 << std::endl;
   std::cout << "summary of sweep optimization up to isweep=" << isweep << std::endl;
   std::cout << tools::line_separator << std::endl;
   std::cout << "comb_schedule: iter, dots, dcut, eps, noise, timing (tsum)" << std::endl;
   std::cout << std::scientific << std::setprecision(2);
   double tsum = 0.0;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto& jctrl = schd.combsweep[jsweep];
      tsum += timing[jsweep];
      std::cout << std::setw(3) << jsweep << " "
           << jctrl.dots << " " 
           << jctrl.dcut << " "
           << jctrl.eps << " " 
           << jctrl.noise << " | "
           << timing[jsweep] << " (" 
           << tsum << ")" << std::endl;
   } // jsweep
   std::cout << "results: dwt, energies (de)" << std::endl;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto dwt = sweep_data[jsweep].first;
      auto& eopt0 = sweep_data[jsweep].second;
      int nstate = eopt0.size();
      std::cout << std::setw(3) << jsweep << " ";
      std::cout << std::showpos << std::scientific << std::setprecision(2) << dwt << std::noshowpos;
      std::cout << std::defaultfloat << std::setprecision(12);
      for(int j=0; j<nstate; j++){ 
         std::cout << " e" << j << ":" 
     	      << std::defaultfloat << std::setprecision(12) << eopt0[j] << " ("
              << std::scientific << std::setprecision(2) << eopt0[j]-eopt[minpos][j] << ")";
      }
      std::cout << std::endl;
   } // jsweep
   std::cout << tools::line_separator2 << std::endl;
}

} // ctns

#endif
