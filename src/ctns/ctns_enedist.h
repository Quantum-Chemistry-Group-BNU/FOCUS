#ifndef CTNS_ENEDIST_H
#define CTNS_ENEDIST_H

#include "sweep_data.h"
#include "sweep_onedot.h"
#include "sweep_twodot.h"
#include "sweep_twodot_enedist.h"
#include "sweep_init.h"
#include "sweep_final.h"
#include "sweep_restart.h"
#include "ctns_outcore.h"
#include "ctns_entropy.h"

namespace ctns{

   // main for sweep optimizations for CTNS
   template <typename Qm, typename Tm>
      sweep_data sweep_enedist(comb<Qm,Tm>& icomb, // initial comb wavefunction
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
            std::cout << "\nctns::sweep_enedist maxsweep=" << schd.ctns.maxsweep << std::endl;
            get_mem_status(rank);
         }
         auto t0 = tools::get_time();

         if(schd.ctns.maxsweep == 0){
            sweep_data sweeps;
            return sweeps;
         }

         // consistency check
         if(schd.ctns.ifdistc && !icomb.topo.ifmps){
            tools::exit("error: ifdistc should be used only with MPS!");
         }
          
         //--------------------------
         // ZL202509: enedist |psi0>
         //--------------------------
         ctns::comb<Qm,Tm> icomb2, licomb2;
         std::vector<qtensor3<Qm::ifabelian,Tm>> cpsis2;
         std::vector<qtensor2<Qm::ifabelian,Tm>> environ(icomb.topo.nbackbone);
         if(schd.ctns.task_enedist){
            if(schd.ctns.rcanon2_file.size()==0){
               tools::exit("error: rcanon2_file must be defined for psi0!");
            }
            if(rank == 0){
               ctns::comb_load(icomb2, schd, schd.ctns.rcanon2_file);
               cpsis2 = ctns::lcanon_canonicalize(licomb2, icomb2, schd, schd.ctns.rcanon2_file, true);           
            } // rank-0 
         }
         //--------------------------

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
            
            // initialize: assume comb is in memory initially
            if(schd.ctns.guess){
               sweep_init(icomb, schd.ctns.nroots, schd.ctns.singlet);
            }

            // outcore: save comb to disk to save memory
            if(schd.ctns.ifoutcore){
               rcanon_save_sites(icomb, scratch, rank);
            }

            //-------------------------------------------------
            // ZL@2025/09: construct environments for MPS only
            //-------------------------------------------------
            if(isweep == schd.ctns.restart_sweep){
               const auto& nodes = icomb.topo.nodes;
               const auto& rindex = icomb.topo.rindex;
               // right boundary
               for(int i=icomb.topo.nbackbone-1; i>0; i--){
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
                  const auto& site2 = icomb2.sites[rindex.at(std::make_pair(i,0))];
                  if(i == icomb.topo.nbackbone-1){
                     environ[i] = contract_qt3_qt3("cr",site,site2);
                  }else{
                     auto qtmp = contract_qt3_qt2("r",site2,environ[i+1]);
                     environ[i] = contract_qt3_qt3("cr",site,qtmp);
                  }
               } // i
               // left boundary
               const auto& site = icomb.sites[rindex.at(std::make_pair(0,0))];
               const auto& site2 = licomb2.sites[rindex.at(std::make_pair(0,0))];
               environ[0] = contract_qt3_qt3("lc",site,site2);
            }
            //-------------------------------------------------
 
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
                  get_mem_status(rank);
               }
               const int pdx = icomb.topo.rindex.at(dbond.get_current());
               const int ndx = icomb.topo.rindex.at(dbond.get_next());
               // for guess 
               if(schd.ctns.ifoutcore){
                  rcanon_load_site(icomb, ndx, scratch, rank); 
               }
         
               if(ibond < schd.ctns.restart_bond){
                  // restart by loading information from disk
                  sweep_restart(icomb, int2e, int1e, ecore, schd, scratch,
                        qops_pool, sweeps, isweep, ibond);
               }else{
                  // ZL@202509: currently only support twodot algorithm
                  assert(dots == 2); 
                  sweep_twodot_enedist(icomb, int2e, int1e, ecore, schd, scratch,
                        qops_pool, sweeps, isweep, ibond, 
                        icomb2, licomb2, cpsis2, environ);
               }
               
               // save updated sites
               if(schd.ctns.ifoutcore){
                  rcanon_save_site(icomb, pdx, scratch, rank);
               }
#ifdef TCMALLOC
   	         release_freecpumem();
#endif
               // timing 
               if(debug){
                  const auto& timing = sweeps.opt_timing[isweep][ibond];
                  sweeps.timing_sweep[isweep].accumulate(timing, "sweep opt", schd.ctns.verbose>0);
                  timing_global.accumulate(timing, "global opt", schd.ctns.verbose>0);
                  get_mem_status(rank, schd.ctns.verbose>1);
               }
               if(schd.ctns.debug_cpumem and rank>0) get_mem_status(rank);
               // stop just for debug
               if(isweep == schd.ctns.maxsweep-1 && ibond == schd.ctns.maxbond){
                  qops_pool.finalize();
                  if(rank == 0) std::cout << "maxbond reached: exit for debugging sweep_enedist!" << std::endl;
                  icomb.world.barrier();
                  exit(1);
               }
            } // ibond
            if(debug){
               auto tf = tools::get_time();
               sweeps.t_total[isweep] = tools::get_duration(tf-ti);
               sweeps.t_inter[isweep] = oper_timer.sigma.t_inter_tot + oper_timer.renorm.t_inter_tot;
               sweeps.t_gemm[isweep]  = oper_timer.sigma.t_gemm_tot  + oper_timer.renorm.t_gemm_tot;
               sweeps.t_red[isweep]   = oper_timer.sigma.t_red_tot   + oper_timer.renorm.t_red_tot;
               sweeps.summary(isweep, size);
               get_mem_status(rank, schd.ctns.verbose>1);
            }
          
            // finalize: load all sites to memory, as they will be save and checked in sweep_final 
            if(schd.ctns.ifoutcore){
	            rcanon_load_sites(icomb, scratch, rank);
	         }
            // generate right rcanonical form and save checkpoint file
            sweep_final(icomb, schd, scratch, isweep, rcfprefix);
            // compute Hmat for checking purpose 
            oper_final(icomb, int2e, int1e, ecore, schd, scratch, qops_pool, isweep);
            
	         if(debug){
               auto tl = tools::get_time();
               tools::timing("sweep_enedist: isweep="+std::to_string(isweep), ti, tl);
            }
            // post processing
            if(rank == 0){
               if(schd.ctns.task_sdiag){
                  ctns::rcanon_Sdiag_sample(icomb, schd.ctns.iroot, schd.ctns.nsample, 
                        schd.ctns.pthrd, schd.ctns.nprt, schd.ctns.saveconfs);
               }
               if(schd.ctns.task_schmidt){
                  auto schmidt_file = schd.scratch+"/"+rcfprefix+"svalues_isweep"+std::to_string(isweep);
                  ctns::rcanon_schmidt(icomb, schd.ctns.iroot, schmidt_file, schd.ctns.save_schmidt);
               }
	            std::cout << std::endl;
            }
         } // isweep
         qops_pool.finalize();

         // Note: no need to load sites to memory again to output final icomb, because
         // in the last step, rcanon_load_sites has already been called,
         // such that all the sites are in memory.
         //if(schd.ctns.ifoutcore) rcanon_load_sites(icomb, scratch, rank);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::sweep_enedist", t0, t1);
            get_mem_status(rank, schd.ctns.verbose>1);
         }
         return sweeps;
      }

} // ctns

#endif
