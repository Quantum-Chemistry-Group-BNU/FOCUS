#include "../settings/global.h"
#include "../core/dvdson.h"
#include "tns_oper.h"
#include "tns_opt.h"
#include "tns_ham.h"
#include "tns_decimation.h"

using namespace std;
using namespace linalg;
using namespace tns;

// sweep optimizations for Comb
void tns::opt_sweep(const input::schedule& schd,
	            comb& icomb,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
	            const double ecore){
   cout << "\ntns::opt_sweep maxsweep=" << schd.maxsweep << endl;
 
   if(schd.maxsweep == 0) return;

   // prepare environmental operators 
   oper_env_right(icomb, icomb, int2e, int1e, schd.scratch);

   // init left boundary site
   icomb.lsites[make_pair(0,0)] = icomb.get_lbsite();

   auto sweeps = icomb.get_sweeps();
   vector<pair<double,vector<double>>> sweep_data(schd.maxsweep); // (dwt,eopt)
   for(int isweep=0; isweep<schd.maxsweep; isweep++){
      
      auto& ctrl = schd.combsweep[isweep];
      cout << endl;
      input::combsweep_print(ctrl);
      
      int size = sweeps.size();
      vector<double> dwt(size);
      vector<vector<double>> eopt(size);
      vector<int> deff(size);
      // loop over sites
      for(int i=0; i<size; i++){
	 auto dbond = sweeps[i];
         auto p0 = get<0>(dbond);
	 auto p1 = get<1>(dbond);
         auto forward = get<2>(dbond);
         auto p = forward? p0 : p1;
	 int tp0 = icomb.type[p0];
	 int tp1 = icomb.type[p1];
         cout << "\n" << global::line_separator << endl;
         cout << "isweep=" << isweep 
	      << " ibond=" << i << " bond=" 
	      << "(" << p0.first << "," << p0.second << ")"
	      << "[" << icomb.topo[p0.first][p0.second] << "]-"
	      << "(" << p1.first << "," << p1.second << ")"
	      << "[" << icomb.topo[p1.first][p1.second] << "]"
	      << " fw=" << forward
	      << " tp=[" << tp0 << "," << tp1 << "]"
	      << " update=" << !forward //-th site of bond get updated
	      << endl;
         cout << global::line_separator << endl;
	 if(ctrl.dots == 1 || (ctrl.dots == 2 && tp0 == 3 && tp1 == 3)){ 
	    opt_onedot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, 
		       eopt[i], dwt[i], deff[i]);
	 }else{
	    opt_twodot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, 
		       eopt[i], dwt[i], deff[i]);
	 }
      } // i
      
      // print summary 
      cout << "\n" << global::line_separator << endl;
      input::combsweep_print(ctrl);
      cout << global::line_separator << endl;
      vector<double> emean(size,0.0);
      for(int i=0; i<size; i++){
	 auto dbond = sweeps[i];
         auto p0 = get<0>(dbond);
	 auto p1 = get<1>(dbond);
         auto forward = get<2>(dbond);
         auto p = forward? p0 : p1;
         cout << "i=" << i << " bond=" 
	      << "(" << p0.first << "," << p0.second << ")"
	      << "[" << icomb.topo[p0.first][p0.second] << "]-"
	      << "(" << p1.first << "," << p1.second << ")"
	      << "[" << icomb.topo[p1.first][p1.second] << "]"
	      << " fw=" << forward
	      << " deff=" << deff[i]
	      << " dwt=" << showpos << scientific << setprecision(2) << dwt[i] << noshowpos;
	 // print energy
	 cout << defaultfloat << setprecision(12);
	 int nstate = eopt[i].size();
	 for(int j=0; j<nstate; j++){ 
	    cout << "  " << j << ":" << eopt[i][j];
	    emean[i] += eopt[i][j];
	 }
	 emean[i] /= nstate;
	 cout << endl;
      }
      auto pos = std::min_element(emean.begin(), emean.end());
      auto min = std::distance(emean.begin(), pos);
      cout << "min energies at: " << min << endl;
      sweep_data[isweep].first = dwt[min];
      sweep_data[isweep].second = eopt[min];
      cout << global::line_separator << endl;

      cout << "\n" << global::line_separator2 << endl;
      cout << "summary of sweep optimization up to isweep=" << isweep << endl;
      cout << global::line_separator << endl;
      cout << "comb_schedule: iter, dots, dcut, eps, noise, dwts, energies" << endl;
      for(int jsweep=0; jsweep<=isweep; jsweep++){
         auto dwt = sweep_data[jsweep].first;
         auto& eopt0 = sweep_data[jsweep].second;
         int nstate = eopt0.size();
	 auto& jctrl = schd.combsweep[jsweep];
	 cout << setw(3) << jctrl.isweep << " "
	      << jctrl.dots << " " 
	      << jctrl.dcut << " "
	      << scientific << setprecision(1) << jctrl.eps << " " 
	      << scientific << setprecision(1) << jctrl.noise << " | " 
              << "dwt="  << showpos << scientific << setprecision(2) << dwt << noshowpos;
         cout << defaultfloat << setprecision(12);
         for(int j=0; j<nstate; j++){ 
            cout << " e" << j << ":" 
		 << defaultfloat << setprecision(12) << eopt0[j] << " "
	         << scientific << setprecision(2) << eopt0[j]-eopt[min][j];
         }
         cout << endl;
      } // jsweep
      cout << global::line_separator2 << endl;
   } // isweep
}

void tns::opt_onedot(const input::schedule& schd,
		     const input::sweep_ctrl& ctrl,
                     comb& icomb,
		     directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore,
		     vector<double>& eopt,
		     double& dwt,
		     int& deff){
   auto t0 = global::get_time();
   cout << "tns::opt_onedot";

   // 0. determine bond and site to be updated
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   
   // 1. process symmetry information & operators for {|lcr>}
   qsym_space qc, ql, qr;
   oper_dict cqops, lqops, rqops;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
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
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   cout << " dim=" << wf.get_dim() << info << endl;
   auto ta = global::get_time();

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = schd.nroots;
   auto diag = tns::get_onedot_Hdiag(cqops, lqops, rqops, int2e, ecore, wf);
   auto tb = global::get_time();

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
   // solve
   eopt.resize(neig);
   matrix vsol(nsub,neig);
   //solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   if(icomb.psi0.size() == 0){
      solver.solve_iter(eopt.data(), vsol.data());
   }else{
      cout << "initial guess" << endl;
      //solver.solve_iter(eopt.data(), vsol.data(), v0.data());
      exit(1);
   }
   auto tc = global::get_time();
   
   // 3. decimation & renormalize operators
   decimation_onedot(icomb, p, forward, cturn, ctrl.dcut, vsol, wf, dwt, deff,
		     ctrl.noise, cqops, lqops, rqops);
   auto td = global::get_time();

   oper_renorm_onedot(icomb, p, forward, cturn, 
		      cqops, lqops, rqops, int2e, int1e, 
		      schd.scratch);
   auto t1 = global::get_time();

   cout << "timing for tns::opt_onedot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
   opt_timing({t0,ta,tb,tc,td,t1});
}

void tns::opt_twodot(const input::schedule& schd,
		     const input::sweep_ctrl& ctrl,
                     comb& icomb,
		     directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore,
		     vector<double>& eopt,
		     double& dwt,
		     int& deff){
   auto t0 = global::get_time();
   cout << "tns::opt_twodot";

   // 0. determine bond and site to be updated
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   
   // 1. process symmetry information & operators for {|lmvr>}
   qsym_space qc1, qc2, ql, qr;
   oper_dict c1qops, c2qops, lqops, rqops;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
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
   auto ta = global::get_time();

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = schd.nroots;
   auto diag = tns::get_twodot_Hdiag(c1qops, c2qops, lqops, rqops, int2e, ecore, wf);
   auto tb = global::get_time();
   
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
   //solver.solve_diag(eopt.data(), vsol.data(), true); // debug
   if(icomb.psi0.size() == 0){
      solver.solve_iter(eopt.data(), vsol.data());
   }else{
      cout << "initial guess" << endl;
      //solver.solve_iter(eopt.data(), vsol.data(), v0.data());
      exit(1);
   }
   auto tc = global::get_time();
   
   // 3. decimation & renormalize operators
   decimation_twodot(icomb, p, forward, cturn, ctrl.dcut, vsol, wf, dwt, deff);
   auto td = global::get_time();
   
   oper_renorm_twodot(icomb, p, forward, cturn, 
		      c1qops, c2qops, lqops, rqops, int2e, int1e, 
		      schd.scratch);
   auto t1 = global::get_time();

   cout << "timing for tns::opt_twodot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
   opt_timing({t0,ta,tb,tc,td,t1});
}

void tns::opt_timing(const vector<tns::tm> ts){
   double dt0 = global::get_duration(ts[1]-ts[0]);
   double dt1 = global::get_duration(ts[2]-ts[1]);
   double dt2 = global::get_duration(ts[3]-ts[2]);
   double dt3 = global::get_duration(ts[4]-ts[3]);
   double dt4 = global::get_duration(ts[5]-ts[4]);
   double dt  = global::get_duration(ts[5]-ts[0]); // total
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
