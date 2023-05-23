#ifndef OPER_ENV_H
#define OPER_ENV_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include "oper_dot.h"
#include "oper_io.h"
#include "oper_renorm.h"
#include "oper_pool.h"

namespace ctns{

   // initialization of operators for 
   // (1) dot operators [c] 
   // (2) boundary operators [l/r]
   template <typename Km>
      void oper_init_dotAll(const comb<Km>& icomb,
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const std::string scratch,
            const int iomode,
            const bool ifdist1){
         using Tm = typename Km::dtype;
         const int isym = Km::isym;
         const bool ifkr = Km::ifkr;
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
               int kp = node.pindex;
               oper_dict<Tm> qops;
               oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank, ifdist1);
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
               int kp = node.pindex;
               oper_dict<Tm> qops;
               oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank, ifdist1);
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

         // lop: left boundary at the start (0,0)
         auto p = std::make_pair(0,0);
         auto ta = tools::get_time();
         //---------------------------------------------
         int kp = icomb.topo.get_node(p).pindex;
         oper_dict<Tm> qops;
         oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size, rank, ifdist1);
         //---------------------------------------------
         auto tb = tools::get_time();
         //---------------------------------------------
         std::string fop = oper_fname(scratch, p, "l");
         oper_save(iomode, fop, qops, debug);
         //---------------------------------------------
         auto tc = tools::get_time();
         t_comp += tools::get_duration(tb-ta);
         t_save += tools::get_duration(tc-tb);

         auto t1 = tools::get_time();
         if(debug){
            tools::timing("ctns::oper_init", t0, t1);
            std::cout << "T[ctns::oper_init](comp/save/tot)="
               << t_comp << "," 
               << t_save << ","
               << (t_comp + t_save)
               << std::endl;
         }
      }

   // build right environment operators
   template <typename Km>
      void oper_env_right(const comb<Km>& icomb, 
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const input::schedule& schd,
            const std::string scratch){
         using Tm = typename Km::dtype;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif   
         const auto& iomode = schd.ctns.iomode;
         const auto& ifdist1 = schd.ctns.ifdist1;
         const bool debug = (rank==0);
         if(debug){ 
            std::cout << "\nctns::oper_env_right qkind=" << qkind::get_name<Km>() << std::endl;
         }
         double t_init = 0.0, t_load = 0.0, t_comp = 0.0, t_save = 0.0;

         // 1. construct for dot [cop] & boundary operators [lop/rop]
         auto t0 = tools::get_time();
         oper_init_dotAll(icomb, int2e, int1e, scratch, iomode, ifdist1);
         auto ta = tools::get_time();
         t_init = tools::get_duration(ta-t0);

         // 2. successive renormalization process
         oper_timer.sweep_start();
         dot_timing timing_sweep, timing;
         oper_pool<Tm> qops_pool(iomode, debug && schd.ctns.verbose>1);
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto p = icomb.topo.rcoord[idx];
            const auto& node = icomb.topo.get_node(p);
            if(node.type != 0 || p.first == 0){
               auto tb = tools::get_time();
               timing.t0 = tools::get_time();
               if(debug) std::cout << "\nidx=" << idx << " coord=" << p << std::endl;

               // a. get operators from memory / disk    
               std::vector<std::string> fneed(2);
               fneed[0] = icomb.topo.get_fqop(p, "c", scratch);
               fneed[1] = icomb.topo.get_fqop(p, "r", scratch);
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

               // b. perform renormalization for superblock {|cr>}
               std::string frop = oper_fname(scratch, p, "r");
               std::string superblock = "cr";
               std::string fname;
               if(schd.ctns.save_formulae) fname = scratch+"/rformulae_env_idx"
                  + std::to_string(idx) + ".txt";
               std::string fmmtask;
               if(debug && schd.ctns.save_mmtask){
                  fmmtask =  "rmmtasks_gemm_idx"+std::to_string(idx);
               }
               oper_renorm(superblock, icomb, p, int2e, int1e, schd,
                     cqops, rqops, qops_pool[frop], fname, timing, fmmtask); 
               auto td = tools::get_time();
               t_comp += tools::get_duration(td-tc);
               timing.tf = tools::get_time();

               // c. save operators to disk
               qops_pool.join_and_erase(fneed);
               qops_pool.save_to_disk(frop, schd.ctns.alg_renorm>10 && schd.ctns.async_tocpu, schd.ctns.async_save);
               auto te = tools::get_time();
               t_save += tools::get_duration(te-td);
               timing.t1 = tools::get_time();

               if(debug){ 
                  timing.analysis("local", schd.ctns.verbose>0);
                  timing_sweep.accumulate(timing, "sweep", schd.ctns.verbose>0);
               }
            }
         } // idx
         qops_pool.finalize();

         auto t1 = tools::get_time();
         if(debug){
            tools::timing("ctns::oper_env_right", t0, t1);
            std::cout << "T[ctns::oper_env_right](init/load/comp/save/tot)="
               << t_init << "," << t_load << "," 
               << t_comp << "," << t_save << ","
               << (t_init + t_load + t_comp + t_save)
               << std::endl;
         }
      }

} // ctns

#endif
