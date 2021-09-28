#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_twodot_renorm.h"
#include "sweep_twodot_hdiag.h"
//#include "sweep_twodot_local.h"
//#include "sweep_twodot_sigma.h"

namespace ctns{

//
// twodot optimization algorithm
//
template <typename Km>
void sweep_twodot(const input::schedule& schd,
		  sweep_data& sweeps,
		  const int isweep,
		  const int ibond,
                  comb<Km>& icomb,
                  const integral::two_body<typename Km::dtype>& int2e,
                  const integral::one_body<typename Km::dtype>& int1e,
                  const double ecore){
   const bool debug_sweep = (schd.ctns.verbose > 0);
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::sweep_twodot" << std::endl;
   const int isym = Km::isym;
   const bool ifkr = qkind::is_kramers<Km>();
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. check partition
   const auto& dbond = sweeps.seq[ibond];
   const auto& p0 = dbond.p0;
   const auto& p1 = dbond.p1;
   const auto& p = dbond.p;
   const auto& cturn = dbond.cturn;
   std::vector<int> suppc1, suppc2, suppl, suppr;
   if(rank == 0 && debug_sweep) std::cout << "support info:" << std::endl;
   if(!cturn){
      //
      //       |    |
      //    ---p0---p1---
      //
      suppc1 = icomb.topo.get_suppc(p0, rank == 0 && debug_sweep); 
      suppc2 = icomb.topo.get_suppc(p1, rank == 0 && debug_sweep); 
      suppl  = icomb.topo.get_suppl(p0, rank == 0 && debug_sweep);
      suppr  = icomb.topo.get_suppr(p1, rank == 0 && debug_sweep);
   }else{
      //       |
      //    ---p1
      //       |
      //    ---p0---
      //
      suppc1 = icomb.topo.get_suppc(p1, rank == 0 && debug_sweep); 
      suppc2 = icomb.topo.get_suppr(p1, rank == 0 && debug_sweep); 
      suppl  = icomb.topo.get_suppl(p0, rank == 0 && debug_sweep);
      suppr  = icomb.topo.get_suppr(p0, rank == 0 && debug_sweep);
   }
   int sc1 = suppc1.size();
   int sc2 = suppc2.size();
   int sl = suppl.size();
   int sr = suppr.size();
   assert(sc1+sc2+sl+sr == icomb.topo.nphysical);

   // 1. load operators 
   using Tm = typename Km::dtype;
   oper_dict<Tm> lqops, rqops, c1qops, c2qops;
   if(!cturn){
      oper_load_qops(icomb, p0, schd.scratch, "l", lqops );
      oper_load_qops(icomb, p1, schd.scratch, "r", rqops );  
      oper_load_qops(icomb, p0, schd.scratch, "c", c1qops);
      oper_load_qops(icomb, p1, schd.scratch, "c", c2qops);
   }else{
      oper_load_qops(icomb, p0, schd.scratch, "l", lqops );
      oper_load_qops(icomb, p0, schd.scratch, "r", rqops );  
      oper_load_qops(icomb, p1, schd.scratch, "c", c1qops);
      oper_load_qops(icomb, p1, schd.scratch, "r", c2qops);
   }
   if(rank == 0){
      std::cout << "qops info: rank=" << rank << std::endl;
      const int level = 0;
      lqops.print("lqops", level);
      rqops.print("rqops", level);
      c1qops.print("c1qops", level);
      c2qops.print("c2qops", level);
   }
   timing.ta = tools::get_time();

   // 2. twodot wavefunction
   //	 \ /
   //   --*--
   const auto& ql = lqops.qbra;
   const auto& qr = rqops.qbra;
   const auto& qc1 = c1qops.qbra;
   const auto& qc2 = c2qops.qbra;
   if(rank == 0){
      if(debug_sweep) std::cout << "qbond info:" << std::endl;
      ql.print("ql", debug_sweep);
      qr.print("qr", debug_sweep);
      qc1.print("qc1", debug_sweep);
      qc2.print("qc2", debug_sweep);
   }
   auto sym_state = get_qsym_state(isym, schd.nelec, schd.twoms);
   stensor4<Tm> wf(sym_state, ql, qr, qc1, qc2);
   if(rank == 0 && debug_sweep){ 
      std::cout << "sym_state=" << sym_state << " dim(localCI)=" << wf.size() << std::endl;
   }
 
   // 3. Davidson solver for wf
   int nsub = wf.size();
   int neig = sweeps.nroots;
   auto& nmvp = sweeps.opt_result[isweep][ibond].nmvp;
   auto& eopt = sweeps.opt_result[isweep][ibond].eopt;
   linalg::matrix<Tm> vsol(nsub,neig);
   
   // 3.1 Hdiag 
   std::vector<double> diag(nsub,1.0);
   twodot_Hdiag(ifkr, lqops, rqops, c1qops, c2qops, ecore, wf, diag, size, rank);
#ifndef SERIAL
   // reduction of partial Hdiag: no need to broadcast, if only rank=0 
   // executes the preconditioning in Davidson's algorithm
   if(size > 1){
      std::vector<double> diag2(nsub);
      boost::mpi::reduce(icomb.world, diag, diag2, std::plus<double>(), 0);
      diag = std::move(diag2);
   }
#endif 
   timing.tb = tools::get_time();

/*
   // 3.2 Solve local problem: Hc=cE
   auto Hx_funs = twodot_Hx_functors(isym, ifkr, lqops, rqops, c1qops, c2qops, 
	                             int2e, int1e, wf, size, rank);
   using std::placeholders::_1;
   using std::placeholders::_2;
   auto HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2,
                    std::ref(wf), std::ref(Hx_funs),
		    std::cref(ifkr), std::cref(ecore), 
		    std::cref(size), std::cref(rank));
   oper_timer.clear();
   twodot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, dbond, wf);
   timing.tc = tools::get_time();
   if(rank == 0){ 
      sweeps.print_eopt(isweep, ibond);
      oper_timer.analysis();
   }
*/

   // 3. decimation & renormalize operators
   twodot_renorm(sweeps, isweep, ibond, icomb, vsol, wf, 
		 lqops, rqops, c1qops, c2qops, int2e, int1e, schd.scratch);

   timing.t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::sweep_twodot", timing.t0, timing.t1);
      timing.analysis();
   }
}

} // ctns

#endif
