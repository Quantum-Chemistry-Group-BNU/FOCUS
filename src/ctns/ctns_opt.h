#ifndef CTNS_OPT_H
#define CTNS_OPT_H

#include "../io/input.h"

namespace ctns{

// main for sweep optimizations for CTNS
template <typename Km>
void opt_sweep(comb<Km>& icomb, // initial comb wavefunction
	       const integral::two_body<typename Km::dtype>& int2e,
	       const integral::one_body<typename Km::dtype>& int1e,
	       const double ecore,
	       const input::schedule& schd){
   auto t0 = tools::get_time();
   std::cout << "\nctns::opt_sweep maxsweep=" << schd.maxsweep << std::endl;

   if(schd.maxsweep == 0) return;
   
   // prepare environmental operators 
   oper_env_right(icomb, int2e, int1e, schd.scratch);
/*
   // init left boundary site
   icomb.lsites[std::make_pair(0,0)] = get_left_bsite<Tm>();
   // generate sweeps
   auto sweeps = icomb.topo.get_sweeps();
   std::vector<dot_result> sweep_summary(schd.maxsweep); 
   std::vector<double> timing(schd.maxsweep);
   for(int isweep=0; isweep<schd.maxsweep; isweep++){
      std::cout << std::endl;
      // sweep control
      auto& ctrl = schd.combsweep[isweep];
      input::combsweep_print(ctrl);
      // data per sweep
      int size = sweeps.size();
      std::vector<dot_result> sweep_result(size);
      // loop over sites
      auto ti = tools::get_time();
      for(int i=0; i<size; i++){
	 auto dbond = sweeps[i];
         auto p0 = std::get<0>(dbond);
	 auto p1 = std::get<1>(dbond);
         auto forward = std::get<2>(dbond);
	 auto updated = !forward; // 0/1-th udpated site in the bond
         auto p = forward? p0 : p1;
	 auto tp0 = icomb.topo.node_type(p0);
	 auto tp1 = icomb.topo.node_type(p1);
         bool cturn = (tp0 == 3 && p1.second == 1);
	 std::cout << "isweep=" << isweep << " ibond=" << i << " bond=" << p0 << "-" << p1 
	           << " (fw,ct,update)=(" << forward << "," << cturn << "," << updated << ")" 
	           << std::endl;
	 sweep_result[i].eopt.resize(icomb.get_nstate());
	 if(ctrl.dots == 1 || (ctrl.dots == 2 && tp0 == 3 && tp1 == 3)){
	    //opt_sweep_onedot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, sweep_result[i]);
	 }else{
	    //opt_sweep_twodot(schd, ctrl, icomb, dbond, int2e, int1e, ecore, sweep_result[i]);
	 }
      } // i
      auto tf = tools::get_time();
      timing[isweep] = tools::get_duration(tf-ti);
      opt_sweep_summary(schd, sweeps, sweep_result, isweep, timing, sweep_summary);
   } // isweep
   //opt_finaldot(schd, icomb, int2e, int1e, ecore);
*/
   auto t1 = tools::get_time();
   std::cout << "\ntiming for ctns::opt_sweep : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

} // ctns

#endif
