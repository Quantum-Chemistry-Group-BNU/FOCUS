#include "../settings/global.h"
#include "../core/dvdson.h"
#include "tns_oper.h"
#include "tns_opt.h"
#include "tns_ham.h"
#include "tns_decimation.h"

#include "../core/linalg.h"

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
   icomb.lsites[make_pair(0,0)] = icomb.get_lbsite();

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
         cout << "\n### isweep=" << isweep << " i=" << i << " bond=" 
	      << "(" << p0.first << "," << p0.second << ")"
	      << "[" << icomb.topo[p0.first][p0.second] << "]-"
	      << "(" << p1.first << "," << p1.second << ")"
	      << "[" << icomb.topo[p1.first][p1.second] << "]"
	      << " forward=" << forward
	      << " type=[" << icomb.type[p0] << "," << icomb.type[p1] << "]"
	      << " updated site=(" << p.first << "," << p.second << ")"
	      << endl;
	 
	 opt_onedot(schd, icomb, dbond, int2e, int1e, ecore);
	 //opt_twodot(schd, icomb, dbond, int2e, int1e, ecore);

	 //if(isweep==1 && i==0){
         //   cout << "break here" << endl;
	 //   exit(1);
	 //}

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
   
   // 1. process symmetry information & operators for {|lcr>}
   qsym_space qc, ql, qr;
   qc = icomb.get_qc(p); 
   ql = icomb.get_ql(p);
   qr = icomb.get_qr(p);
   
   // wavefunction to be computed
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   wf.print("wf",1);

   oper_dict cqops, lqops, rqops;
   cqops = oper_get_cqops(icomb, p, schd.scratch);
   lqops = oper_get_lqops(icomb, p, schd.scratch);
   rqops = oper_get_rqops(icomb, p, schd.scratch);
   cout << "c:" << oper_dict_opnames(cqops) << " "   
	<< "l:" << oper_dict_opnames(lqops) << " " 
	<< "r:" << oper_dict_opnames(rqops) << " "  
	<< endl;

   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_onedot_Hdiag(cqops, lqops, rqops, ecore, wf);

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
  
   // initial guess 
   matrix v0 = random_matrix(nsub, neig);
   vector<double> v0tmp(nsub*neig);
   copy(v0.data(), v0.data()+nsub*neig, v0tmp.begin());
   int nindp = linalg::get_ortho_basis(nsub,neig,v0tmp);
   assert(nindp == neig);
   copy(v0tmp.begin(), v0tmp.end(), v0.data());

   // solve
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   //solver.solve_iter(esol.data(), vsol.data(), v0.data());
   solver.solve_diag(esol.data(), vsol.data());

   wf.from_array(vsol.data());
   cout << wf.normF() << endl;
   cout << "energy=" << esol[0] << endl;

   int Dcut = 2;
   // 3. decimation & renormalize operators
   oper_dict qops;
   bool cturn = (icomb.type[p0] == 3 && p1.second == 1);
   if(forward){
      if(!cturn){
         // update lsites & ql
         cout << "renormlize |lc>" << endl;
	 auto qt2 = wf.merge_lc();
	 qt2 = decimation_row(qt2, Dcut);
         qsym_space_print(qt2.qcol, "after renormalization");
	 icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lc()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lc(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
	 qops = oper_renorm_ops("lc",icomb, icomb, p, lqops, cqops, int2e, int1e);
      }else{
	 // update lsites & qc [special for comb]
         cout << "renormlize |lr>" << endl;
	 assert(p0.second == 0);
	 auto qt2 = wf.merge_lr();
	 qt2 = decimation_row(qt2, Dcut);
         qsym_space_print(qt2.qcol, "after renormalization");
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
 	 //-------------------------------------------------------------------	 
	 assert((qt2-icomb.lsites[p].merge_lr()).normF() < 1.e-10);
	 auto ovlp = contract_qt3_qt3_lr(icomb.lsites[p],icomb.lsites[p]);
	 assert(ovlp.check_identity(1.e-10,false)<1.e-10);
 	 //-------------------------------------------------------------------	 
	 qops = oper_renorm_ops("lr",icomb, icomb, p, lqops, rqops, int2e, int1e);
      }
      string fname = oper_fname(schd.scratch, p, "lop");
      oper_save(fname, qops);
   }else{
      // update rsites (p1) & qr
      cout << "renormlize |cr>" << endl;
      auto qt2 = wf.merge_cr();
      qt2 = decimation_col(qt2, Dcut);
      qsym_space_print(qt2.qrow, "after renormalization");
      icomb.rsites[p] = qt2.split_cr(wf.qmid, wf.qcol, wf.dpt_cr().second);
      //-------------------------------------------------------------------	
      assert((qt2-icomb.rsites[p].merge_cr()).normF() < 1.e-10);	 
      auto ovlp = contract_qt3_qt3_cr(icomb.rsites[p],icomb.rsites[p]);
      assert(ovlp.check_identity(1.e-10,false)<1.e-10);
      //-------------------------------------------------------------------	 
      qops = oper_renorm_ops("cr",icomb, icomb, p, cqops, rqops, int2e, int1e);
      string fname = oper_fname(schd.scratch, p, "rop");
      oper_save(fname, qops);
   }

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
   cout << "tns::opt_twodot" << endl;

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
      //c1qops = oper_get_cqops(icomb, p0, schd.scratch);
      //c2qops = oper_get_cqops(icomb, p1, schd.scratch);
      //lqops  = oper_get_lqops(icomb, p0, schd.scratch);
      //rqops  = oper_get_rqops(icomb, p1, schd.scratch);  
   }else{
      qc1 = icomb.get_qc(p1);
      qc2 = icomb.get_qr(p1);
      ql  = icomb.get_ql(p0);
      qr  = icomb.get_qr(p0);
      //c1qops = oper_get_cqops(icomb, p1, schd.scratch);
      //c2qops = oper_get_rqops(icomb, p1, schd.scratch);
      //lqops  = oper_get_lqops(icomb, p0, schd.scratch);
      //rqops  = oper_get_rqops(icomb, p0, schd.scratch);  
   }
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor4 wf(sym_state,qc1,qc2,ql,qr);
   wf.print("wf",1);

/*
   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_twodot_Hdiag(c1qops, c2qops, lqops, rqops, ecore, wf);
   
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
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
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   solver.solve_iter(esol.data(), vsol.data());
   cout << "energy=" << esol[0] << endl;
   exit(1);
*/
   //test decimation
   wf.random();
   cout << wf.normF() << endl;
   double nrm = wf.normF();
   wf *= 1.0/nrm;
   cout << wf.normF() << endl;

   int Dcut = 20;
   // 3. decimation & renormalize operators
   if(forward){
      if(!cturn){
         // update lsites & ql
         cout << "renormlize |lc1>" << endl;
	 auto qt2 = wf.merge_lc1().merge_cr();
	 qt2 = decimation_row(qt2, Dcut);
         qsym_space_print(qt2.qcol, "after renormalization");
         icomb.lsites[p] = qt2.split_lc(wf.qrow, wf.qmid, wf.dpt_lc1().second);
         //oper_renorm_ops("lc",icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }else{
	 // update lsites & qc [special for comb]
         cout << "renormlize |lr>" << endl;
	 assert(p0.second == 0);
	 auto qt2 = wf.merge_lr_c1c2();
	 qt2 = decimation_row(qt2, Dcut);
         qsym_space_print(qt2.qcol, "after renormalization");
	 icomb.lsites[p]= qt2.split_lr(wf.qrow, wf.qcol, wf.dpt_lr().second);
         //oper_renorm_ops("lr",icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }
   }else{
      if(!cturn){
         // update rsites (p1) & qr
         cout << "renormlize |c2r>" << endl;
         auto qt2 = wf.merge_c2r().merge_lc();
         qt2 = decimation_col(qt2, Dcut);
         qsym_space_print(qt2.qrow, "after renormalization");
         icomb.rsites[p] = qt2.split_cr(wf.qver, wf.qcol, wf.dpt_c2r().second);
         //oper_renorm_ops("cr",icomb,icomb,p0,int2e,int1e,schd.scratch);
      }else{
	 // update rsites & qr [special for comb]
         cout << "renormlize |c1r2>" << endl;
	 assert(p0.second == 0);
	 auto qt2 = wf.merge_lr_c1c2();
	 qt2 = decimation_col(qt2, Dcut);
         qsym_space_print(qt2.qrow, "after renormalization");
	 icomb.rsites[p]= qt2.split_cr(wf.qmid, wf.qver, wf.dpt_c1c2().second);
         //oper_renorm_ops("lr",icomb,icomb,p0,int2e,int1e,schd.scratch); ???
      }
   }

   auto t1 = global::get_time();
   cout << "timing for tns::opt_twodot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
