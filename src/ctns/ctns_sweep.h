#ifndef CTNS_SWEEP_H
#define CTNS_SWEEP_H

#include "sweep_data.h"
#include "sweep_rcanon.h"
#include "sweep_restart.h"
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
         if(schd.ctns.maxsweep == 0) return;
         auto t0 = tools::get_time();

         // global timer
         dot_timing timing_global;
         oper_timer.sweep_start();
         // generate sweep sequence
         auto sweep_seq = icomb.topo.get_sweeps(debug);
         sweep_data sweeps(sweep_seq, schd.ctns.nroots, schd.ctns.maxsweep, 
               schd.ctns.restart_sweep, schd.ctns.ctrls);
         // pool for handling operators
         oper_pool<Tm> qops_pool(schd.ctns.iomode, debug);
         for(int isweep=0; isweep<schd.ctns.maxsweep; isweep++){
            if(isweep < schd.ctns.restart_sweep) continue; // restart case
            if(debug){
               std::cout << tools::line_separator2 << std::endl;
               sweeps.print_ctrls(isweep); // print sweep control
               std::cout << tools::line_separator2 << std::endl;
            }
            // initialize
            if(rank == 0) sweep_init(icomb, schd.ctns.nroots);
            // loop over sites
            auto ti = tools::get_time();
            for(int ibond=0; ibond<sweeps.seqsize; ibond++){
               const auto& dbond = sweeps.seq[ibond];
               const auto& dots = sweeps.ctrls[isweep].dots;
               auto tp0 = icomb.topo.get_type(dbond.p0);
               auto tp1 = icomb.topo.get_type(dbond.p1);
               if(debug){
                  std::cout << "\nisweep=" << isweep 
                     << " ibond=" << ibond << "/seqsize=" << sweeps.seqsize
                     << " dots=" << dots << " dbond=" << dbond
                     << std::endl;
                  std::cout << tools::line_separator << std::endl;
               }
               if(ibond < schd.ctns.restart_bond){
                  sweep_restart(icomb, int2e, int1e, ecore, schd, scratch,
                        qops_pool, sweeps, isweep, ibond);
               }else{
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
                     sweeps.timing_sweep[isweep].accumulate(timing, "sweep", schd.ctns.verbose>0);
                     timing_global.accumulate(timing, "global", schd.ctns.verbose>0);
                  }
               }
               // stop just for debug
               if(isweep==schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond) exit(1);
            } // ibond
            if(debug){
               auto tf = tools::get_time();
               sweeps.t_total[isweep] = tools::get_duration(tf-ti);
               sweeps.t_inter[isweep] = oper_timer.sigma.t_inter_tot + oper_timer.renorm.t_inter_tot;
               sweeps.t_gemm[isweep] = oper_timer.sigma.t_gemm_tot + oper_timer.renorm.t_gemm_tot;
               sweeps.t_red[isweep] = oper_timer.sigma.t_red_tot + oper_timer.renorm.t_red_tot;
               sweeps.summary(isweep);
            }
            // generate right rcanonical form and save checkpoint file
            if(rank == 0) sweep_final(icomb, schd, scratch, isweep);
         } // isweep
         qops_pool.finalize();

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::sweep_opt", t0, t1);
            if(schd.ctns.verbose>0) timing_global.print("global");
         }
      }

   } // ctns

#endif
