#ifndef CTNS_SWEEP_H
#define CTNS_SWEEP_H

#include "sweep_data.h"

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
   auto sweep_seq = icomb.topo.get_sweeps();
   sweep_data sdata(schd, sweep_seq.size());
   for(int isweep=0; isweep<schd.maxsweep; isweep++){
      // sweep control
      std::cout << std::endl;
      auto& ctrl = sdata.ctrls[isweep];
      input::combsweep_print(ctrl);
      
      // loop over sites
      auto ti = tools::get_time();
      for(int ibond=0; ibond<sdata.seqsize; ibond++){
	 auto dbond = sweep_seq[ibond];
         auto p0 = std::get<0>(dbond);
	 auto p1 = std::get<1>(dbond);
         auto forward = std::get<2>(dbond);
	 auto updated = !forward; // 0/1-th udpated site in the bond
         auto p = forward? p0 : p1;
	 auto tp0 = icomb.topo.node_type(p0);
	 auto tp1 = icomb.topo.node_type(p1);
         bool cturn = icomb.topo.is_cturn(p0,p1);
	 std::cout << " isweep=" << isweep << " ibond=" << ibond << " bond=" << p0 << "-" << p1 
	           << " (forward,cturn,update)=(" << forward << "," << cturn << "," << updated << ")" 
	           << std::endl;

/*
	 sweep_result[i].eopt.resize(icomb.get_nstate());
	 if(ctrl.dots == 1 || (ctrl.dots == 2 && tp0 == 3 && tp1 == 3)){
	    //opt_sweep_onedot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, sweep_result[i]);
	 }else{
	    //opt_sweep_twodot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, sweep_result[i]);
	 }
*/
      } // i
      auto tf = tools::get_time();
/*
      timing[isweep] = tools::get_duration(tf-ti);
      opt_sweep_summary(schd, sweep_seq, sweep_result, isweep, timing, sweep_summary);
*/
   } // isweep
   //opt_finaldot(schd, icomb, int2e, int1e, ecore);

   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::opt_sweep : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
   exit(1);
}

} // ctns

#endif
