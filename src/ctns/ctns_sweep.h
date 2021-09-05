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
	       const input::schedule& schd){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0){ 
      std::cout << "\nctns::sweep_opt maxsweep=" << schd.ctns.maxsweep << std::endl;
   }
   auto t0 = tools::get_time();

   if(schd.ctns.maxsweep == 0) return;

   // prepare environmental operators 
   auto Hmat = get_Hmat(icomb, int2e, int1e, ecore, schd.scratch);
   if(rank == 0) Hmat.print("Hmat",8);

   // init left boundary site
   get_left_bsite(Km::isym, icomb.lsites[std::make_pair(0,0)]);

   // generate sweep sequence
   sweep_data sweeps(icomb.topo.get_sweeps(rank==0), 
		     schd.ctns.nroots, schd.ctns.guess, 
		     schd.ctns.maxsweep, schd.ctns.ctrls, schd.ctns.dbranch,
		     schd.ctns.rdm_vs_svd); 
   for(int isweep=0; isweep<schd.ctns.maxsweep; isweep++){
      // print sweep control
      if(rank == 0){
         std::cout << tools::line_separator2 << std::endl;
         sweeps.print_ctrls(isweep);
         std::cout << tools::line_separator2 << std::endl;
      }
      // loop over sites
      auto ti = tools::get_time();
      for(int ibond=0; ibond<sweeps.seqsize; ibond++){
	 auto dbond = sweeps.seq[ibond];
         auto p0 = dbond.p0;
	 auto p1 = dbond.p1;
         auto forward = dbond.forward;
         auto p = dbond.p;
         bool cturn = dbond.cturn;
	 auto dots = sweeps.ctrls[isweep].dots;
#ifndef SERIAL
	 icomb.world.barrier();
#endif
	 if(rank == 0){
	    std::cout << "\nisweep=" << isweep << " ibond=" << ibond << " bond=" << p0 << "-" << p1 
	              << " (dots,forward,cturn)=(" << dots << "," << forward << "," << cturn << ")" 
	              << std::endl;
            std::cout << tools::line_separator << std::endl;
	 }
	 auto tp0 = icomb.topo.node_type(p0);
	 auto tp1 = icomb.topo.node_type(p1);
	 if(dots == 1){ // || (dots == 2 && tp0 == 3 && tp1 == 3)){
	    sweep_onedot(schd, sweeps, isweep, ibond, icomb, int2e, int1e, ecore);
	 }else{
	    sweep_twodot(schd, sweeps, isweep, ibond, icomb, int2e, int1e, ecore);
	 }
      } // ibond
      auto tf = tools::get_time();
      sweeps.t_total[isweep] = tools::get_duration(tf-ti);
      if(rank == 0) sweeps.summary(isweep);
   } // isweep

   // get rwfuns, which is useful for later computing properties
   sweep_rwfuns(schd, icomb, int2e, int1e, ecore);

   auto t1 = tools::get_time();
   if(rank == 0) tools::timing("ctns::opt_sweep", t0, t1);
}

} // ctns

#endif
