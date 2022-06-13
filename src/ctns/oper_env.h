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
   const auto& alg_renorm = schd.ctns.alg_renorm;
   const auto& save_formulae = schd.ctns.save_formulae;
   const auto& sort_formulae = schd.ctns.sort_formulae;
   const auto& ifdist1 = schd.ctns.ifdist1;
   const bool debug = (rank==0);
   if(debug){ 
      std::cout << "\nctns::oper_env_right Km=" << qkind::get_name<Km>() << std::endl;
   }
   double t_init = 0.0, t_load = 0.0, t_comp = 0.0, t_save = 0.0;

   // 1. construct for dot [cop] & boundary operators [lop/rop]
   auto t0 = tools::get_time();
   oper_init_dotAll(icomb, int2e, int1e, scratch, iomode, ifdist1);
   auto ta = tools::get_time();
   t_init = tools::get_duration(ta-t0);

   // 2. successive renormalization process
   oper_pool<Tm> qops_pool(iomode, schd.ctns.ioasync, debug);
   for(int idx=0; idx<icomb.topo.ntotal; idx++){
      auto p = icomb.topo.rcoord[idx];
      const auto& node = icomb.topo.get_node(p);
      if(node.type != 0 || p.first == 0){
         auto tb = tools::get_time();
         if(debug){ 
	    std::cout << "\nidx=" << idx 
		      << " coord=" << p 
		      << std::endl;
         }
         // a. get operators from memory / disk    
         std::vector<std::string> fneed(2);
	 fneed[0] = icomb.topo.get_fqop(p, "c", scratch);
	 fneed[1] = icomb.topo.get_fqop(p, "r", scratch);
         qops_pool.fetch(fneed);
	 const auto& cqops = qops_pool(fneed[0]);
	 const auto& rqops = qops_pool(fneed[1]);
         auto tc = tools::get_time();
         t_load += tools::get_duration(tc-tb); 
         // b. perform renormalization for superblock {|cr>}
	 std::string frop = oper_fname(scratch, p, "r");
	 std::string superblock = "cr";
         std::string fname;
	 if(save_formulae) fname = scratch+"/rformulae_env_idx"
		   		 + std::to_string(idx) + ".txt"; 
         oper_renorm_opAll(superblock, icomb, p, int2e, int1e,
			   cqops, rqops, qops_pool(frop), 
			   fname, alg_renorm, sort_formulae, ifdist1);
         auto td = tools::get_time();
         t_comp += tools::get_duration(td-tc);
         // c. save operators to disk
         qops_pool.save(frop);
         auto te = tools::get_time();
	 t_save += tools::get_duration(te-td);
      }
   } // idx
   qops_pool.clean_up();
 
   auto t1 = tools::get_time();
   if(debug){
      tools::timing("ctns::oper_env_right", t0, t1);
      std::cout << "T[ctns::oper_env_right](init/load/comp/save/tot)="
                << t_init << "," 
		<< t_load << "," 
		<< t_comp << "," 
		<< t_save << ","
                << (t_init + t_load + t_comp + t_save)
                << std::endl;
   }
}

} // ctns

#endif
