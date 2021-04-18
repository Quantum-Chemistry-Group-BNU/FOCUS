#ifndef CTNS_SWEEP_H
#define CTNS_SWEEP_H

#include "sweep_data.h"
#include "sweep_util.h"

namespace ctns{

// main for sweep optimizations for CTNS
template <typename Km>
void sweep_opt(comb<Km>& icomb, // initial comb wavefunction
	       const integral::two_body<typename Km::dtype>& int2e,
	       const integral::one_body<typename Km::dtype>& int1e,
	       const double ecore,
	       const input::schedule& schd){
   auto t0 = tools::get_time();
   std::cout << "\nctns::sweep_opt maxsweep=" << schd.maxsweep << std::endl;

   if(schd.maxsweep == 0) return;
   
   // prepare environmental operators 
   oper_env_right(icomb, int2e, int1e, schd.scratch);

   // init left boundary site
   get_left_bsite(Km::isym, icomb.lsites[std::make_pair(0,0)]);

   // generate sweep sequence
   sweep_data sweeps(schd, icomb.topo.get_sweeps(), icomb.get_nstate());
   for(int isweep=0; isweep<schd.maxsweep; isweep++){
      // print sweep control
      sweeps.print_ctrl(isweep);
      // loop over sites
      auto ti = tools::get_time();
      for(int ibond=0; ibond<sweeps.seqsize; ibond++){
	 auto dbond = sweeps.seq[ibond];
         auto p0 = dbond.p0;
	 auto p1 = dbond.p1;
         auto forward = dbond.forward;
         auto p = dbond.p;
         bool cturn = dbond.cturn;
	 std::cout << " isweep=" << isweep << " ibond=" << ibond << " bond=" << p0 << "-" << p1 
	           << " (forward,update,cturn)=(" << forward << "," << !forward << "," << cturn << ")" 
	           << std::endl;

	 auto tp0 = icomb.topo.node_type(p0);
	 auto tp1 = icomb.topo.node_type(p1);
	 if(sweeps.ctrls[isweep].dots == 1){ // || (ctrl.dots == 2 && tp0 == 3 && tp1 == 3)){
	    sweep_onedot(schd, sweeps, isweep, ibond, icomb, int2e, int1e, ecore);
	 }else{
	    //sweep_twodot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, sweep_result[i]);
	 }

      } // i
      auto tf = tools::get_time();
      sweeps.timing[isweep] = tools::get_duration(tf-ti);
      sweeps.summary(isweep);
   } // isweep
   //opt_finaldot(schd, icomb, int2e, int1e, ecore);

   auto t1 = tools::get_time();
   std::cout << "timing for ctns::opt_sweep : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // ctns

#endif
