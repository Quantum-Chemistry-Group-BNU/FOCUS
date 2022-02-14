#ifndef OPER_ENV_H
#define OPER_ENV_H

#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

#include "oper_dict.h"
#include "oper_dot.h"
#include "oper_io.h"
#include "oper_renorm.h"

namespace ctns{

// initialization of operators for 
// (1) dot operators [c] 
// (2) boundary operators [l/r]
template <typename Km>
void oper_init_dotAll(const comb<Km>& icomb,
         	      const integral::two_body<typename Km::dtype>& int2e,
         	      const integral::one_body<typename Km::dtype>& int1e,
         	      const std::string scratch){
   using Tm = typename Km::dtype;
   const int isym = Km::isym;
   const bool ifkr = qkind::is_kramers<Km>();
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif  
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
	 oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size);
	 //---------------------------------------------
         auto tb = tools::get_time();
	 //---------------------------------------------
	 std::string fname = oper_fname(scratch, p, "c");
         oper_save(fname, qops);
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
	 oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size);
	 //---------------------------------------------
         auto tb = tools::get_time();
	 //---------------------------------------------
	 std::string fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
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
   oper_init_dot(qops, isym, ifkr, kp, int2e, int1e, size);
   //---------------------------------------------
   auto tb = tools::get_time();
   //---------------------------------------------
   std::string fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
   //---------------------------------------------
   auto tc = tools::get_time();
   t_comp += tools::get_duration(tb-ta);
   t_save += tools::get_duration(tc-tb);

   auto t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::oper_init", t0, t1);
      std::cout << "detailed T(comp/save)= "
	        << t_comp << ", " << t_save
                << " T(total)=" << (t_comp + t_save) << " S"
                << std::endl;
   }
}

// build right environment operators
template <typename Km>
void oper_env_right(const comb<Km>& icomb, 
		    const integral::two_body<typename Km::dtype>& int2e,
		    const integral::one_body<typename Km::dtype>& int1e,
		    const std::string scratch,
		    const int algorithm){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0){ 
      std::cout << "\nctns::oper_env_right Km=" << qkind::get_name<Km>() << std::endl;
   }
   double t_init = 0.0, t_load = 0.0, t_comp = 0.0, t_save = 0.0;
   
   auto t0 = tools::get_time();
   //---------------------------------------------
   // construct for dot operators [cop] & boundary operators [lop/rop]
   oper_init_dotAll(icomb, int2e, int1e, scratch);
   //---------------------------------------------
   auto ta = tools::get_time();
   t_init = tools::get_duration(ta-t0);

   // successive renormalization process
   for(int idx=0; idx<icomb.topo.ntotal; idx++){
      auto p = icomb.topo.rcoord[idx];
      const auto& node = icomb.topo.get_node(p);
      if(node.type != 0 || p.first == 0){
         auto tb = tools::get_time();
         //---------------------------------------------
         // load operators from disk    
         //---------------------------------------------
	 oper_dict<typename Km::dtype> qops1, qops2, qops;
	 //std::cout << "load1" << std::endl;
         oper_load_qops(icomb, p, scratch, "c", qops1);
	 //qops1.print("qops1");
	 //std::cout << "data=" << qops1._data << std::endl; 
	 //std::cout << "load2" << std::endl;
         oper_load_qops(icomb, p, scratch, "r", qops2);
	 //qops2.print("qops2");
	 //std::cout << "data=" << qops2._data << std::endl; 
         //---------------------------------------------
         auto tc = tools::get_time();
         t_load += tools::get_duration(tc-tb); 
         //---------------------------------------------
         // perform renormalization for superblock {|cr>}
         //---------------------------------------------
	 std::string superblock = "cr"; 
         oper_renorm_opAll(superblock, icomb, p, int2e, int1e,
			   qops1, qops2, qops, algorithm);
         //qops.print("qops");
         //std::cout << "data=" << qops._data << std::endl; 
         //---------------------------------------------
         auto td = tools::get_time();
         t_comp += tools::get_duration(td-tc);
         //---------------------------------------------
	 // save operators to disk
         //---------------------------------------------
	 std::string fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
         //---------------------------------------------
         auto te = tools::get_time();
	 t_save += tools::get_duration(te-td);
      }
   } // idx

   auto t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::oper_env_right", t0, t1);
      std::cout << "detailed T(init/load/comp/save)= "
                << t_init << ", " << t_load << ", " << t_comp << ", " << t_save
                << " T(total)=" << (t_init + t_load + t_comp + t_save) << " S"
                << std::endl;
   }
}

} // ctns

#endif
