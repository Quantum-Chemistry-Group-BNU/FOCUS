#ifndef CTNS_SWEEP_H
#define CTNS_SWEEP_H

#include "sweep_data.h"
#include "sweep_onedot.h"
#include "sweep_twodot.h"

namespace ctns{

// main for sweep optimizations for CTNS
template <typename Km>
void sweep_opt(comb<Km>& icomb, // initial comb wavefunction
	       const integral::two_body<typename Km::dtype>& int2e,
	       const integral::one_body<typename Km::dtype>& int1e,
	       const double ecore,
	       const input::schedule& schd,
	       const std::string scratch){
   using Tm = typename Km::dtype;
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif  
   const bool debug = (rank==0); 
   if(debug){ 
      std::cout << "\nctns::sweep_opt maxsweep=" 
	        << schd.ctns.maxsweep 
		<< std::endl;
   }
   auto t0 = tools::get_time();

   if(schd.ctns.maxsweep == 0) return;

   // init left boundary site
   const auto& ntotal = icomb.topo.ntotal;
   icomb.lsites.resize(ntotal);
   icomb.lsites[ntotal-1] = get_left_bsite<Tm>(Km::isym);

   // generate sweep sequence
   sweep_data sweeps(icomb.topo.get_sweeps(rank==0), 
		     schd.ctns.nroots, schd.ctns.guess, schd.ctns.maxsweep, 
		     schd.ctns.ctrls, schd.ctns.dbranch, schd.ctns.rdm_vs_svd);
   oper_pool<Tm> qops_pool(schd.ctns.iomode, schd.ctns.ioasync, debug);
   for(int isweep=0; isweep<schd.ctns.maxsweep; isweep++){
      // print sweep control
      if(debug){
         std::cout << tools::line_separator2 << std::endl;
         sweeps.print_ctrls(isweep);
         std::cout << tools::line_separator2 << std::endl;
      }
      // loop over sites
      auto ti = tools::get_time();
      for(int ibond=0; ibond<sweeps.seqsize; ibond++){
	 const auto& dbond = sweeps.seq[ibond];
	 const auto& dots = sweeps.ctrls[isweep].dots;
	 auto tp0 = icomb.topo.get_type(dbond.p0);
	 auto tp1 = icomb.topo.get_type(dbond.p1);
#ifndef SERIAL
	 icomb.world.barrier();
#endif
	 if(debug){
	    std::cout << "\nisweep=" << isweep 
                      << " ibond=" << ibond << "/seqsize=" << sweeps.seqsize
		      << " dots=" << dots << " dbond=" << dbond
	              << std::endl;
            std::cout << tools::line_separator << std::endl;
	 }
	 // optimization
	 if(dots == 1){ // || (dots == 2 && tp0 == 3 && tp1 == 3)){
	    sweep_onedot(icomb, int2e, int1e, ecore, schd, scratch,
			 qops_pool, sweeps, isweep, ibond); 
	 }else{
	    sweep_twodot(icomb, int2e, int1e, ecore, schd, scratch,
			 qops_pool, sweeps, isweep, ibond); 
	 }
	 // timing 
         if(debug){
            const auto& timing = sweeps.opt_timing[isweep][ibond];
	    sweeps.timing_sweep[isweep].accumulate(timing,"time_sweep");
            sweeps.timing_global.accumulate(timing,"time_global");
         }
         // just for debug
	 if(isweep == schd.ctns.maxsweep-1 && ibond == schd.ctns.maxbond) exit(1);
      } // ibond
      auto tf = tools::get_time();
      sweeps.t_total[isweep] = tools::get_duration(tf-ti);
      if(debug) sweeps.summary(isweep);
   } // isweep

   // for later computing properties
   if(schd.ctns.lastdot) sweep_rwfuns(icomb, int2e, int1e, ecore, 
		   		      schd, scratch, qops_pool);
   qops_pool.clean_up();

   auto t1 = tools::get_time();
   if(debug) tools::timing("ctns::opt_sweep", t0, t1);
}

} // ctns

#endif
