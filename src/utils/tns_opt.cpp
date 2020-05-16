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
   //oper_env_right(icomb, icomb, int2e, int1e, schd.scratch);

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
         cout << "\ni=" << i << " bond=" 
	      << "(" << p0.first << "," << p0.second << ")-"
	      << "(" << p1.first << "," << p1.second << ")"
	      << " forward=" << forward
	      << " type=[" << icomb.type[p0] << "," << icomb.type[p1] << "]"
	      << " updated=(" << p.first << "," << p.second << ")"
	      << endl;
	 opt_onedot(schd, icomb, dbond, int2e, int1e, ecore);
	 //opt_twodot(schd, icomb, dbond, int2e, int1e, ecore);
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
   
   // 1. process symmetry information & operators for {|lcr>}
   auto qc = icomb.get_qc(p); 
   auto ql = icomb.get_ql(p);
   auto qr = icomb.get_qr(p);
   oper_dict cqops, lqops, rqops;
   //cqops = oper_get_cqops(icomb, p, schd.scratch);
   //lqops = oper_get_lqops(icomb, p, schd.scratch);
   //rqops = oper_get_rqops(icomb, p, schd.scratch);
   // wavefunction to be computed
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   wf.print("wf",1);

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   //auto diag = tns::get_onedot_Hdiag(cqops, lqops, rqops, ecore, wf);
   vector<double> diag(nsub,1.0);

   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
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
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   solver.solve_iter(esol.data(), vsol.data());
   cout << "energy=" << esol[0] << endl;

   // test
   wf.random();
   cout << wf.normF() << endl;
   double nrm = wf.normF();
   wf *= 1.0/nrm;
   cout << wf.normF() << endl;

   int Dcut = 10;
   // 3. decimation & renormalize operators
   if(!forward){
      // update rsites (p1) & qr
      cout << "renormlize |cr>" << endl;
      auto qt2 = wf.merge_cr();
      qt2 = decimation_col(qt2, Dcut);
      auto dpt = wf.dpt_cr().second;
      auto qt3 = qt2.split_cr(wf.qmid,wf.qcol,dpt);
      icomb.rsites[p] = qt3;
      //oper_renorm_rops(icomb,icomb,p0,int2e,int1e,schd.scratch);
   }else{
      // update lsites (p0)
      if(p1.second == 1){
	 assert(p0.second == 0);
	 // update lsites & qc [special for comb]
         cout << "renormlize |lr>" << endl;
	 auto qt2 = wf.merge_lr();
	 qt2 = decimation_row(qt2, Dcut);
         auto dpt = wf.dpt_lr().second;
	 auto qt3 = qt2.split_lr(wf.qrow,wf.qcol,dpt);
         icomb.lsites[p] = qt3;
         //oper_renorm_lops(icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }else{
         // update lsites & ql
         cout << "renormlize |lc>" << endl;
	 auto qt2 = wf.merge_lc();
	 qt2 = decimation_row(qt2, Dcut);
         auto dpt = wf.dpt_lc().second;
	 auto qt3 = qt2.split_lc(wf.qrow,wf.qmid,dpt);
         icomb.lsites[p] = qt3;
         //oper_renorm_lops(icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }
   }
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
   auto pm = icomb.get_c(p1);
   auto pv = icomb.get_r(p1);
   auto pr = icomb.get_r(p0);

   // 1. process symmetry information & operators
   auto ql = icomb.get_ql(p0);
   auto qm = icomb.get_qc(p1); 
   auto qv = icomb.get_qr(p1);
   auto qr = icomb.get_qr(p0);
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor4 wf(sym_state,qm,qv,ql,qr);
   wf.print("wf",1);
   wf.random();
   cout << wf.get_dim() << endl;

/*
   qsym_space qlc1, qc2r;
   qsym_dpt dpt1, dpt2;
   qsym_space_dpt(ql,qm,qlc1,dpt1);
   qsym_space_dpt(qv,qr,qc2r,dpt2);

   auto qt3 = merge_qt4_qt3_lc1(wf,qlc1,dpt1);
   cout << endl;
   qt3.print("qt3",1);
   cout << qt3.get_dim() << endl;
   cout << "lzdA" << endl;

   auto qt4 = split_qt4_qt3_lc1(qt3,ql,qm,dpt1);
   cout << endl;
   qt4.print("qt4",1);
   cout << qt4.get_dim() << endl;
   cout << "lzdB" << endl;
   
   auto qt3x = merge_qt4_qt3_c2r(wf,qc2r,dpt2);
   cout << "lzdC" << endl;
   cout << endl;
   qt3x.print("qt3x",1);
   cout << qt3.get_dim() << endl;
   
   auto qt4x = split_qt4_qt3_c2r(qt3x,qv,qr,dpt2);
   cout << "lzdD" << endl;
   cout << endl;
   qt4x.print("qt4x",1);
   cout << qt4.get_dim() << endl;
   
   cout << "lzdE" << endl;
   qsym_space qlr, qc1c2;
   qsym_dpt dptx, dpty;
   qsym_space_dpt(ql,qr,qlr,dptx);
   qsym_space_dpt(qm,qv,qc1c2,dpty);
   auto qt2xy = merge_qt4_qt2_lr_c1c2(qt4,qlr,qc1c2,dptx,dpty);
   qt2xy.print("qt2xy",1);
   cout << qt2xy.get_dim_row() << endl;
   cout << qt2xy.get_dim_col() << endl;
   cout << qt2xy.get_dim() << endl;

   cout << "---qt3qt2---" << endl;
   qsym_space qlc, qcr;
   qsym_dpt dpta, dptb;
   qsym_space_dpt(ql,qm,qlc1,dpt1);
   qsym_space_dpt(qv,qr,qc2r,dpt2);

   auto qt2 = merge_qt3_qt2_lc(qt3x,qlc1,dpt1);
   qt2.print("qt2",1);
   cout << qt2.get_dim() << endl;

   auto qt3a = split_qt3_qt2_lc(qt2,ql,qm,dpt1);
   qt3a.print("qt3a",1);
   cout << qt3a.get_dim() << endl;

   auto qt2x = merge_qt3_qt2_cr(qt3,qc2r,dpt2);
   qt2x.print("qt2x",1);
   cout << qt2x.get_dim() << endl;

   auto qt3b = split_qt3_qt2_cr(qt2x,qv,qr,dpt2);
   qt3b.print("qt3b",1);
   cout << qt3b.get_dim() << endl;

   auto qt3s = split_qt3_qt2_lr(qt2xy,ql,qr,dptx);
   qt3s.print("qt3s",1);
   cout << qt3s.get_dim() << endl;

   auto qt2m = merge_qt3_qt2_lr(qt3s,qlr,dptx);
   qt2m.print("qt2m",1);
   cout << qt2m.get_dim() << endl;
   cout << "diff=" << (qt2m-qt2xy).normF() << endl;
   cout << "normF=" << qt2m.normF() << endl;
    
   // test decimation in various directions

   // expand into sites

   exit(1);
*/
   auto lqops = oper_get_lqops(icomb, p0, schd.scratch);
   auto cqops = oper_get_cqops(icomb, p1, schd.scratch);
   auto vqops = oper_get_rqops(icomb, p1, schd.scratch);
   auto rqops = oper_get_rqops(icomb, p0, schd.scratch);
  
   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_twodot_Hdiag(cqops, vqops, lqops, rqops, ecore, wf);
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
/*
   solver.HVec = bind(tns::get_twodot_Hx, _1, _2, 
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
   exit(1);

   auto t1 = global::get_time();
   cout << "timing for tns::opt_twodot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
