#include "../settings/global.h"
#include "../core/dvdson.h"
#include "tns_oper.h"
#include "tns_opt.h"

using namespace std;
using namespace linalg;
using namespace tns;

// sweep optimizations for Comb
// see my former implementation of DMRG

void tns::opt_sweep(const input::schedule& schd,
	            comb& icomb,
	            const integral::two_body& int2e,
	            const integral::one_body& int1e,
	            const double ecore){
   cout << "\ntns::opt_sweep" << endl;
   
   // prepare environmental operators 
   oper_env_right(icomb, icomb, int2e, int1e, schd.scratch);
   
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
         cout << "i=" << i << " : " 
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
   cout << "tns::opt_onedot" << endl;

   // 1. process symmetry information
   qsym_space ql;
   qsym_space qc;
   qsym_space qr;
   //qtensor3 qt(ql,qc,qr);

   exit(1);

   //int nsub = qt.get_dim();
   int nsub = 0;
   int neig = 1;

   //auto diag = tns::get_Hdiag();

   // 2. Davidson solver 
   dvdsonSolver solver;
   solver.iprt = 2;
   solver.crit_v = schd.crit_v;
   solver.maxcycle = schd.maxcycle;
   solver.ndim = nsub;
   solver.neig = neig;
   //solver.Diag = diag.data();
   using std::placeholders::_1;
   using std::placeholders::_2;
   //solver.HVec = bind(tns::get_Hx, _1, _2);
   // solve
   vector<double> esol(neig);
   matrix vsol(nsub,neig);
   //solver.solve_iter(esol.data(), vsol.data());

   cout << "energy=" << esol[0] << endl;

   // 3. decimation
   //qtensor3 site = tns::decimation(icomb, dbond, vsol);

   // 4. renormalize operators
   auto forward = get<2>(dbond);
   if(forward){
      // TRY TO GETRID OF p0
      //auto& site = icomb.lsites[p]; 
      //oper_renorm_left(bra,ket,p,p0,int2e,int1e,scratch);
   }else{
      // TRY TO GETRID OF p0
      //auto& site = icomb.rsites[p]; 
      //oper_renorm_right(site,site,p,p0,int2e,int1e,schd.scratch);
   }
   
   auto t1 = global::get_time();
   cout << "\ntiming for tns::opt_onedot : " << setprecision(2) 
        << global::get_duration(t1-t0) << " s" << endl;
}
