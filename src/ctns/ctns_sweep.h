#ifndef CTNS_SWEEP_H
#define CTNS_SWEEP_H

#include "sweep_data.h"
#include "sweep_onedot.h"
#include "sweep_twodot.h"
#include "sweep_init.h"
#include "sweep_final.h"
#include "ctns_outcore.h"

namespace ctns{

   // main for sweep optimizations for CTNS
   template <typename Qm, typename Tm>
      sweep_data sweep_opt(comb<Qm,Tm>& icomb, // initial comb wavefunction
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            const std::string rcfprefix=""){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "\nctns::sweep_opt maxsweep=" << schd.ctns.maxsweep << std::endl;
            get_sys_status();
         }
         auto t0 = tools::get_time();

         if(schd.ctns.maxsweep == 0){
            sweep_data sweeps;
            return sweeps;
         }

         // consistency check
         if(schd.ctns.ifdistc && !icomb.topo.ifmps){
            std::cout << "error: ifdistc should be used only with MPS!" << std::endl;
            exit(1);
         }

         // 0. outcore
         const int rdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
         const int rdx1 = icomb.topo.rindex.at(std::make_pair(1,0));
         if(schd.ctns.ifoutcore){
            rcanon_save_sites(icomb, scratch, debug);
         }

         // build operators on the left dot
         oper_init_dotL(icomb, int2e, int1e, schd, scratch);

         // global timer
         dot_timing timing_global;
         // generate sweep sequence
         auto sweep_seq = icomb.topo.get_sweeps(false,debug);
         sweep_data sweeps(sweep_seq, schd.ctns.nroots, schd.ctns.maxsweep, 
               schd.ctns.restart_sweep, schd.ctns.ctrls);
         // pool for handling operators
         qoper_pool<Qm::ifabelian,Tm> qops_pool(schd.ctns.iomode, debug && schd.ctns.verbose>1);
         for(int isweep=0; isweep<schd.ctns.maxsweep; isweep++){
            if(isweep < schd.ctns.restart_sweep) continue; // restart case
            if(debug){
               std::cout << tools::line_separator2 << std::endl;
               sweeps.print_ctrls(isweep); // print sweep control
               std::cout << tools::line_separator2 << std::endl;
            }
            const auto& dots = sweeps.ctrls[isweep].dots;
            oper_timer.sweep_start(dots);
            
            // initialize
            if(schd.ctns.guess){
               if(schd.ctns.ifoutcore){
                  rcanon_load_site(icomb, rdx0, scratch, debug); 
                  rcanon_load_site(icomb, rdx1, scratch, debug); 
               }
               sweep_init(icomb, schd.ctns.nroots, schd.ctns.singlet);
               if(schd.ctns.ifoutcore){
                  rcanon_save_site(icomb, rdx0, scratch, debug); 
               }
            }
            
            // loop over sites
            auto ti = tools::get_time();
            for(int ibond=0; ibond<sweeps.seqsize; ibond++){
               const auto& dbond = sweeps.seq[ibond];
               const auto tp0 = icomb.topo.get_type(dbond.p0);
               const auto tp1 = icomb.topo.get_type(dbond.p1);
               if(debug){
                  std::cout << "\nisweep=" << isweep 
                     << " ibond=" << ibond << "/seqsize=" << sweeps.seqsize
                     << " dots=" << dots << " dbond=" << dbond
                     << std::endl;
                  std::cout << tools::line_separator << std::endl;
                  get_sys_status();
               }
               const int pdx = icomb.topo.rindex.at(dbond.get_current());
               const int ndx = icomb.topo.rindex.at(dbond.get_next());
               // for guess 
               if(schd.ctns.ifoutcore){
                  rcanon_load_site(icomb, ndx, scratch, debug); 
               }
               // optimization
               if(dots == 1){ // || (dots == 2 && tp0 == 3 && tp1 == 3)){
                  sweep_onedot(icomb, int2e, int1e, ecore, schd, scratch,
                        qops_pool, sweeps, isweep, ibond); 
               }else{
                  sweep_twodot(icomb, int2e, int1e, ecore, schd, scratch,
                        qops_pool, sweeps, isweep, ibond);
               }
               // save updated sites
               if(schd.ctns.ifoutcore){
                  rcanon_save_site(icomb, pdx, scratch, debug);
               }
               // timing 
               if(debug){
                  const auto& timing = sweeps.opt_timing[isweep][ibond];
                  sweeps.timing_sweep[isweep].accumulate(timing, "sweep opt", schd.ctns.verbose>0);
                  timing_global.accumulate(timing, "global opt", schd.ctns.verbose>0);
                  get_sys_status();
               }
               // stop just for debug [done it for rank-0]
               if(rank==0 && isweep==schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond) exit(1);
            } // ibond
            if(debug){
               auto tf = tools::get_time();
               sweeps.t_total[isweep] = tools::get_duration(tf-ti);
               sweeps.t_inter[isweep] = oper_timer.sigma.t_inter_tot + oper_timer.renorm.t_inter_tot;
               sweeps.t_gemm[isweep]  = oper_timer.sigma.t_gemm_tot  + oper_timer.renorm.t_gemm_tot;
               sweeps.t_red[isweep]   = oper_timer.sigma.t_red_tot   + oper_timer.renorm.t_red_tot;
               sweeps.summary(isweep, size);
               get_sys_status();
            }
          
            // finalize: load all sites, as they will be save and checked in sweep_final 
            if(schd.ctns.ifoutcore) rcanon_load_sites(icomb, scratch, debug);
            // generate right rcanonical form and save checkpoint file
            sweep_final(icomb, schd, scratch, isweep, rcfprefix);
            // compute Hmat 
            oper_final(icomb, int2e, int1e, ecore, schd, scratch, qops_pool, isweep);
            // save updated sites, which will be used in sweep_init in the next sweep
            if(schd.ctns.ifoutcore and isweep != schd.ctns.maxsweep-1){  
               rcanon_save_site(icomb, rdx1, scratch, debug); 
               rcanon_save_site(icomb, rdx0, scratch, debug); 
            }
         } // isweep
         qops_pool.finalize();

         // no need to load to memory again to output final icomb, because
         // in the last step, rcanon_load_sites has already been called,
         // such that all the sites are in memory.
         //if(schd.ctns.ifoutcore) rcanon_load_sites(icomb, scratch, debug);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::sweep_opt", t0, t1);
            get_sys_status();
         }
         return sweeps;
      }

} // ctns

#endif
