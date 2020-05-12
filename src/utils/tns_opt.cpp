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

   // init boundary sites
   for(int idx=0; idx<icomb.ntotal; idx++){
      auto p = icomb.rcoord[idx];
      if(icomb.type[p] == 0) icomb.lsites[p] = icomb.get_bsite();
   }

   // one-dot sweep
   const int nsweeps = 2;
   auto sweeps = icomb.get_sweeps();
   for(int isweep=0; isweep<nsweeps; isweep++){
      cout << "\nisweep = " << isweep << endl;
      for(int i=0; i<sweeps.size(); i++){
	 auto dbond = sweeps[i];
         auto coord0  = get<0>(dbond);
	 auto coord1  = get<1>(dbond);
         auto forward = get<2>(dbond);
         cout << "\ni=" << i << " : " 
	      << "(" << coord0.first << "," << coord0.second << ")->"
	      << "(" << coord1.first << "," << coord1.second << ") "
	      << "forward=" << forward 
	      << endl;
	 opt_onedot(schd, icomb, dbond, int2e, int1e, ecore);
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
   auto p = get<0>(dbond);
   auto ip = p.first, jp = p.second;
   auto pl = icomb.get_l(p);
   auto pc = icomb.get_c(p);
   auto pr = icomb.get_r(p);
   cout << "tns::opt_onedot coord=(" << ip << "," << jp << ")"
	<< "[" << icomb.topo[ip][jp] << "]" << endl; 
   cout << "neighbor: "
	<< "pl=(" << pl.first << "," << pl.second << ") " 
        << "pc=(" << pc.first << "," << pc.second << ") " 
        << "pr=(" << pr.first << "," << pr.second << ") " 
	<< endl;
   
   // 1. process symmetry information
   qsym_space ql, qc, qr;
   ql = icomb.lsites[pl].qcol;
   qr = icomb.rsites[pr].qcol;
   if(pc == make_pair(-1,-1)){
      qc = phys_qsym_space;
   }else{
      qc = icomb.rsites[p].qmid;
   }
   int nelec_a = (schd.nelec+schd.twoms)/2;
   qsym sym_state(schd.nelec,nelec_a);
   qtensor3 wf(sym_state,qc,ql,qr,{0,1,1,1});
   wf.print("wf",1);
   cout << wf.get_dim() << endl;
   
   // 2. Davidson solver 
   int nsub = wf.get_dim();
   int neig = 1; //schd.nroots;
   auto diag = tns::get_Hdiag(icomb, p, int2e, int1e, ecore, schd.scratch, wf);
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
		      cref(int2e), cref(int1e), cref(ecore), 
		      cref(schd.scratch),
		      ref(wf));
   // solve
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   solver.solve_iter(esol.data(), vsol.data());
   cout << "energy=" << esol[0] << endl;
   exit(1);

   // 3. decimation
   //qtensor3 site = tns::decimation(icomb, dbond, vsol);

   // 4. renormalize operators
   auto forward = get<2>(dbond);
   if(forward){
      // TRY TO GETRID OF p0
      //auto& site = icomb.lsites[p]; 
      //oper_renorm_left(icomb,icomb,p,p0,int2e,int1e,scratch);
   }else{
      // TRY TO GETRID OF p0
      //auto& site = icomb.rsites[p]; 
      //oper_renorm_right(site,site,p,p0,int2e,int1e,schd.scratch);
   }
  
   exit(1);

   auto t1 = global::get_time();
   cout << "timing for tns::opt_onedot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
