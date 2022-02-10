#ifndef SWEEP_TWODOT_H
#define SWEEP_TWODOT_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#include "sweep_twodot_renorm.h"
#include "sweep_twodot_hdiag.h"
#include "sweep_twodot_local.h"
#include "sweep_twodot_sigma.h"
#include "symbolic_twodot_formulae.h"
#include "symbolic_twodot_sigma.h"

namespace ctns{

// twodot optimization algorithm
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
   auto& timing = sweeps.opt_timing[isweep][ibond];
   timing.t0 = tools::get_time();

   // 0. check partition
   const auto& dbond = sweeps.seq[ibond];
   const auto& p0 = dbond.p0;
   const auto& p1 = dbond.p1;
   const auto& p = dbond.p;
   const auto& cturn = dbond.cturn;
   std::vector<int> suppc1, suppc2, suppl, suppr;
   if(!cturn){
      //
      //       |    |
      //    ---p0---p1---
      //
      suppl  = icomb.topo.get_suppl(p0);
      suppr  = icomb.topo.get_suppr(p1);
      suppc1 = icomb.topo.get_suppc(p0);
      suppc2 = icomb.topo.get_suppc(p1);
   }else{
      //       |
      //    ---p1
      //       |
      //    ---p0---
      //
      suppl  = icomb.topo.get_suppl(p0);
      suppr  = icomb.topo.get_suppr(p0);
      suppc1 = icomb.topo.get_suppc(p1);
      suppc2 = icomb.topo.get_suppr(p1);
   }
   int sc1 = suppc1.size();
   int sc2 = suppc2.size();
   int sl = suppl.size();
   int sr = suppr.size();
   assert(sc1+sc2+sl+sr == icomb.topo.nphysical);
   if(rank == 0 && debug_sweep){
      std::cout << "support info:" << std::endl;
      tools::print_vector(suppl, "suppl");
      tools::print_vector(suppr, "suppr");
      tools::print_vector(suppc1, "suppc1");
      tools::print_vector(suppc2, "suppc2");
   }

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
      lqops.print("lqops", 2);
      rqops.print("rqops", 2);
      c1qops.print("c1qops", 2);
      c2qops.print("c2qops", 2);
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
   twodot_Hdiag(lqops, rqops, c1qops, c2qops, ecore, wf, diag, size, rank);
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

   // 3.2 Solve local problem: Hc=cE
   std::cout << "schd.ctns.algorithm=" << schd.ctns.algorithm << std::endl;
   using std::placeholders::_1;
   using std::placeholders::_2;
   symbolic_task<Tm> H_formulae;
   Hx_functors<Tm> Hx_funs;
   HVec_type<Tm> HVec;
   if(schd.ctns.algorithm == 0){
      Hx_funs = twodot_Hx_functors(lqops, rqops, c1qops, c2qops, 
                                   int2e, int1e, ecore,
              			   wf, size, rank);
      HVec = bind(&ctns::twodot_Hx<Tm>, _1, _2, std::ref(Hx_funs),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }else if(schd.ctns.algorithm == 1){
      H_formulae = symbolic_twodot_formulae(lqops, rqops, c1qops, c2qops, 
		                            int2e, size, rank);
      HVec = bind(&ctns::symbolic_twodot_Hx<Tm>, _1, _2, std::cref(H_formulae),
		  std::cref(lqops), std::cref(rqops), std::cref(c1qops), 
		  std::cref(c2qops), std::cref(ecore),
                  std::ref(wf), std::cref(size), std::cref(rank));
   }
   oper_timer.clear();
   twodot_localCI(icomb, nsub, neig, diag, HVec, eopt, vsol, nmvp,
		  schd.ctns.cisolver, sweeps.guess, sweeps.ctrls[isweep].eps, 
		  schd.ctns.maxcycle, (schd.nelec)%2, dbond, wf);
   timing.tc = tools::get_time();
   if(rank == 0){ 
      sweeps.print_eopt(isweep, ibond);
      oper_timer.analysis();
   }

   //exit(1);
   //if(ibond == 16) exit(1);

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
