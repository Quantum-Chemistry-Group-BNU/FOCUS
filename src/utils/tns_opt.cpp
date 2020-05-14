#include "../settings/global.h"
#include "../core/dvdson.h"
#include "tns_oper.h"
#include "tns_opt.h"
#include "tns_hamiltonian.h"
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
   cout << "\ntns::opt_sweep" << endl;
   
   // prepare environmental operators 
   oper_env_right(icomb, icomb, int2e, int1e, schd.scratch);

   // init left boundary sites
   icomb.lsites[make_pair(0,0)] = icomb.get_bsite();

   // one-dot sweep
   const int nsweeps = 2;
   auto sweeps = icomb.get_sweeps();
   for(int isweep=0; isweep<nsweeps; isweep++){
      cout << "\nisweep = " << isweep << endl;
      for(int i=0; i<sweeps.size(); i++){
	 auto dbond = sweeps[i];
         auto p0 = get<0>(dbond);
	 auto p1 = get<1>(dbond);
         auto forward = get<2>(dbond);
         auto p = forward? p0 : p1;
         cout << "isweep=" << i << " bond=" 
	      << "(" << p0.first << "," << p0.second << ")-"
	      << "(" << p1.first << "," << p1.second << ")"
	      << " forward=" << forward
	      << " type=[" << icomb.type[p0] << "," << icomb.type[p1] << "]"
	      << " updated=(" << p.first << "," << p.second << ")"
	      << endl;
	 //opt_onedot(schd, icomb, dbond, int2e, int1e, ecore);
	 opt_twodot(schd, icomb, dbond, int2e, int1e, ecore);
      } // i
   } // isweep

}

void tns::opt_onedot(const input::schedule& schd,
                     comb& icomb,
		     directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore){
   bool debug = false;
   auto t0 = global::get_time();
   cout << "tns::opt_onedot" << endl;

   // 0. determine bond and site to be updated
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   auto forward = get<2>(dbond);
   auto p = forward? p0 : p1;
   auto tp0 = icomb.type[p0];
   auto tp1 = icomb.type[p1];
   
   // 1. process symmetry information & operators
   auto pc = icomb.get_c(p);
   auto pl = icomb.get_l(p);
   auto pr = icomb.get_r(p);
   auto qc = icomb.get_qc(p); 
   auto ql = icomb.get_ql(p);
   auto qr = icomb.get_qr(p);
   auto cqops = oper_get_cqops(icomb, p0, schd.scratch);
   auto lqops = oper_get_lqops(icomb, p0, schd.scratch);
   auto rqops = oper_get_rqops(icomb, p0, schd.scratch);
   
   // wavefunction to be computed
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   wf.print("wf",1);

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_Hdiag(cqops, lqops, rqops, ecore, wf);
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(tns::get_Hx, _1, _2, 
		      cref(icomb), cref(p),
		      ref(cqops), ref(lqops), ref(rqops), 
		      cref(int2e), cref(int1e), cref(ecore), 
		      ref(wf));
   // solve
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   solver.solve_iter(esol.data(), vsol.data());
   cout << "energy=" << esol[0] << endl;

/*
   auto p = forward? p0 : p1;
   // 3. decimation & renormalize operators
   if(!forward){
      // update rsites (p1) & qr
      icomb.rsites[p1] = decimation_onedot(icomb, p0, wf, vsol);
      oper_renorm_rops(icomb,icomb,p0,int2e,int1e,schd.scratch);
   }else{
      // update lsites (p0)
      if(p1.second == 1){
	 assert(p0.second == 0);
	 // update lsites & qc [special for comb]
         icomb.lsites[p0] = decimation_onedot(icomb, p0, wf, vsol);
         oper_renorm_lops(icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }else{
         // update lsites & ql
         icomb.lsites[p0] = decimation_onedot(icomb, p0, wf, vsol);
         oper_renorm_lops(icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }
   }
*/
   exit(1);

   auto t1 = global::get_time();
   cout << "timing for tns::opt_onedot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}

void tns::opt_twodot(const input::schedule& schd,
                     comb& icomb,
		     directed_bond& dbond,
                     const integral::two_body& int2e,
                     const integral::one_body& int1e,
                     const double ecore){
   bool debug = false;
   auto t0 = global::get_time();
   auto p0 = get<0>(dbond);
   auto p1 = get<1>(dbond);
   cout << "tns::opt_twodot types=" 
	<< icomb.type[p0] << "," << icomb.type[p1] 
	<< endl;
   
   auto pl = icomb.get_l(p0);
   auto pc = icomb.get_c(p1);
   auto pv = icomb.get_r(p1);
   auto pr = icomb.get_r(p0);

   auto ql = icomb.get_ql(p0);
   auto qc = icomb.get_qc(p1); 
   auto qv = icomb.get_qr(p1);
   auto qr = icomb.get_qr(p0);
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor4 wf(sym_state,qc,qv,ql,qr);
   wf.print("wf",1);
   cout << wf.get_dim() << endl;
   exit(1);


/*
   cout << "neighbor: "
        << "pc0=(" << pc0.first << "," << pc0.second << ") " 
        << "pc1=(" << pc1.first << "," << pc1.second << ") " 
	<< "pl0=(" << pl0.first << "," << pl0.second << ") " 
        << "pr1=(" << pr1.first << "," << pr1.second << ") " 
	<< endl;
  
   // 1. process symmetry information & operators
   auto qc = icomb.get_qc(p); 
   auto ql = icomb.get_ql(p);
   auto qr = icomb.get_qr(p);
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   wf.print("wf",1);
   cout << wf.get_dim() << endl;

   auto cqops = oper_get_cqops(icomb, p, schd.scratch);
   auto lqops = oper_get_lqops(icomb, p, schd.scratch);
   auto rqops = oper_get_rqops(icomb, p, schd.scratch);
  
   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_Hdiag(cqops, lqops, rqops, ecore, wf);
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   solver.HVec = bind(tns::get_Hx, _1, _2, 
		      cref(icomb), cref(p),
		      ref(cqops), ref(lqops), ref(rqops), 
		      cref(int2e), cref(int1e), cref(ecore), 
		      ref(wf));
   // solve
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   solver.solve_iter(esol.data(), vsol.data());
   cout << "energy=" << esol[0] << endl;
   exit(1);

   // 3. decimation & renormalize operators
   auto forward = get<2>(dbond);
   if(forward){
      // update lsites
      //icomb.lsites[p] = decimation_onedot(icomb, p, wf, vsol);
      //oper_renorm_lops(icomb,icomb,p,int2e,int1e,schd.scratch);
   }else{
      // update rsites
      //icomb.rsites[p] = decimation_onedot(icomb, p, wf, vsol);
      //oper_renorm_rops(icomb,icomb,p,int2e,int1e,schd.scratch);
   }
*/

   auto t1 = global::get_time();
   cout << "timing for tns::opt_twodot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
