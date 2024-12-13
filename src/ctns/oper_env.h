#ifndef OPER_ENV_H
#define OPER_ENV_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include "oper_io.h"
#include "oper_pool.h"
#include "oper_dot.h"
#include "oper_renorm.h"
#include "sadmrg/oper_dot_su2.h"
#include "ctns_outcore.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void oper_init_dotL(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            const std::string scratch){

         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const auto& iomode = schd.ctns.iomode;
         const bool debug = (rank==0);
         if(debug){
            std::cout << "\nctns::oper_init_dotL isym=" << isym 
               << " ifkr=" << ifkr 
               << " singlet=" << schd.ctns.singlet
               << std::endl;
         }
         auto t0 = tools::get_time();

         // lop: left boundary at the start (0,0)
         auto p = std::make_pair(0,0);
         //---------------------------------------------
         int kp = icomb.topo.get_node(p).porb;
         qoper_dict<Qm::ifabelian,Tm> qops;
         oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank);
         //---------------------------------------------
         std::string fop = oper_fname(scratch, p, "l");
         if(isym == 3 and schd.ctns.singlet){
            qoper_dict<Qm::ifabelian,Tm> qops2;
            oper_init_dotSE(qops, qops2, icomb.get_qsym_state().ts());
            oper_save(iomode, fop, qops2, debug);
         }else{
            oper_save(iomode, fop, qops, debug);
         }
         //---------------------------------------------

         auto t1 = tools::get_time();
         if(debug) tools::timing("ctns::oper_init_dotL", t0, t1);
      } 

   // initialization of operators for
   // (1) dot operators [c]
   // (2) boundary operators [l/r]
   template <typename Qm, typename Tm>
      void oper_init_dotCR(const comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const std::string scratch,
            const int iomode){
         const int isym = Qm::isym;
         const bool ifkr = Qm::ifkr;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool debug = (rank==0);
         double t_comp = 0.0, t_save = 0.0;

         auto t0 = tools::get_time();
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto p = icomb.topo.rcoord[idx];
            const auto& node = icomb.topo.get_node(p);

            // cop: local operators on physical sites
            if(node.type != 3){
               auto ta = tools::get_time();
               //---------------------------------------------
               int kp = node.porb;
               qoper_dict<Qm::ifabelian,Tm> qops;
               oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank);
               //---------------------------------------------
               auto tb = tools::get_time();
               //---------------------------------------------
               std::string fop = oper_fname(scratch, p, "c");
               oper_save(iomode, fop, qops, debug);
               //---------------------------------------------
               auto tc = tools::get_time();
               t_comp += tools::get_duration(tb-ta);
               t_save += tools::get_duration(tc-tb);
            }

            // rop: right boundary (exclude the start point)
            if(node.type == 0 && p.first != 0){
               auto ta = tools::get_time();
               //---------------------------------------------
               int kp = node.porb;
               qoper_dict<Qm::ifabelian,Tm> qops;
               oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank);
               //---------------------------------------------
               auto tb = tools::get_time();
               //---------------------------------------------
               std::string fop = oper_fname(scratch, p, "r");
               oper_save(iomode, fop, qops, debug);
               //---------------------------------------------
               auto tc = tools::get_time();
               t_comp += tools::get_duration(tb-ta);
               t_save += tools::get_duration(tc-tb);
            }
         }

         auto t1 = tools::get_time();
         if(debug){
            tools::timing("ctns::oper_init_dotCR", t0, t1);
            std::cout << "T[ctns::oper_init_dotCR](comp/save/tot)="
               << t_comp << ","
               << t_save << ","
               << (t_comp + t_save)
               << std::endl;
         }
      }

   // build right environment operators
   template <typename Qm, typename Tm>
      void oper_env_right(comb<Qm,Tm>& icomb, // icomb may be modified ifoutcore=true 
            const integral::two_body<Tm>& int2e,
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            const std::string scratch){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif   
         const auto& iomode = schd.ctns.iomode;
         const bool debug = (rank==0);
         if(debug){ 
            std::cout << "\nctns::oper_env_right qkind=" << qkind::get_name<Qm>() << std::endl;
            get_cpumem_status(rank);
         }
         double t_init = 0.0, t_load = 0.0, t_comp = 0.0, t_save = 0.0;
        
         // 0. outcore
         if(schd.ctns.ifoutcore) rcanon_save_sites(icomb, scratch, debug);

         // 1. construct for dot [cop] & boundary operators [lop/rop]
         auto t0 = tools::get_time();
         oper_init_dotCR(icomb, int2e, int1e, scratch, iomode);
         auto ta = tools::get_time();
         t_init = tools::get_duration(ta-t0);

         // 2. successive renormalization process
         oper_timer.sweep_start();
         dot_timing timing_sweep, timing;
         qoper_pool<Qm::ifabelian,Tm> qops_pool(iomode, debug && schd.ctns.verbose>1);
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto pcoord = icomb.topo.rcoord[idx];
            const auto& node = icomb.topo.get_node(pcoord);
            if(node.type != 0 || pcoord.first == 0){
               auto tb = tools::get_time();
               timing.t0 = tools::get_time();
               if(debug){
                  std::cout << "\nidx=" << idx << " coord=" << pcoord << std::endl;
                  get_cpumem_status(rank);
               }
               
               // a. get operators from memory / disk
               std::vector<std::string> fneed(2);
               fneed[0] = icomb.topo.get_fqop(pcoord, "c", scratch);
               fneed[1] = icomb.topo.get_fqop(pcoord, "r", scratch);
               qops_pool.fetch_to_memory(fneed, schd.ctns.alg_renorm>10);
               const auto& cqops = qops_pool.at(fneed[0]);
               const auto& rqops = qops_pool.at(fneed[1]);
               if(debug && schd.ctns.verbose>0){
                  cqops.print("cqops");
                  rqops.print("rqops");
               }
               auto tc = tools::get_time();
               t_load += tools::get_duration(tc-tb);
               timing.ta = tools::get_time();
               timing.tb = timing.ta;
               timing.tc = timing.ta;
               timing.td = timing.ta;
               timing.te = timing.ta;
               
               // ZL@2024/12/08
               if(schd.ctns.ifoutcore) rcanon_load_site(icomb, idx, scratch, debug);

               // b. perform renormalization for superblock {|cr>}
               std::string frop = oper_fname(scratch, pcoord, "r");
               std::string superblock = "cr";
               std::string fname;
               if(schd.ctns.save_formulae) fname = scratch+"/rformulae_env_idx"
                  + std::to_string(idx) + ".txt";
               std::string fmmtask;
               if(debug && schd.ctns.save_mmtask){
                  fmmtask =  "rmmtasks_idx"+std::to_string(idx);
               }
               oper_renorm(superblock, icomb, pcoord, int2e, int1e, schd,
                     cqops, rqops, qops_pool[frop], fname, timing, fmmtask);
               auto td = tools::get_time();
               t_comp += tools::get_duration(td-tc);

               qops_pool.clear_from_memory({fneed[0],fneed[1]}, {});
               timing.tf14 = tools::get_time();
               
               if(schd.ctns.ifab2pq){
                  const int nsite = icomb.get_nphysical();
                  const bool ifmps = icomb.topo.ifmps;
                  const bool ab2pq_current = get_ab2pq_current(superblock, ifmps, nsite, pcoord, schd.ctns.ifab2pq, 2);
                  if(ab2pq_current) oper_ab2pq(superblock, icomb, pcoord, int2e, schd, qops_pool[frop]);
               }
               timing.tf = tools::get_time();
               
               // ZL@2024/12/08
               if(schd.ctns.ifoutcore) rcanon_clear_site(icomb, idx);

               // c. save operators to disk
               qops_pool.join_and_erase(fneed);
               qops_pool.save_to_disk(frop, schd.ctns.async_save);
               auto te = tools::get_time();
               t_save += tools::get_duration(te-td);
               timing.t1 = tools::get_time();

#ifdef TCMALLOC
   	         release_freecpumem();
#endif
               if(debug){ 
                  get_cpumem_status(rank);
                  timing.analysis("local oper_env", schd.ctns.verbose>0);
                  timing_sweep.accumulate(timing, "sweep oper_env", schd.ctns.verbose>0);
               }
               // stop just for debug
               if(idx == schd.ctns.maxidx){
                  qops_pool.finalize();
                  if(rank == 0) std::cout << "exit for debug oper_env" << std::endl;
                  exit(1);
               }
            }
         } // idx
         qops_pool.finalize();
         
         // 3. outcore
         if(schd.ctns.ifoutcore) rcanon_load_sites(icomb, scratch, debug);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::oper_env_right", t0, t1);
            std::cout << "T[ctns::oper_env_right](init/load/comp/save/tot)="
               << t_init << "," << t_load << "," 
               << t_comp << "," << t_save << ","
               << (t_init + t_load + t_comp + t_save)
               << std::endl;
            get_cpumem_status(rank, schd.ctns.verbose>1);
         }
      }

} // ctns

#endif
