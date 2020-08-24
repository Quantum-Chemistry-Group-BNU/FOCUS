#ifndef CTNS_OPER_HELPER_H
#define CTNS_OPER_HELPER_H

#include "ctns_io.h"
#include "ctns_comb.h"
#include "ctns_oper_util.h"
#include "ctns_oper_dot.h"
#include "ctns_oper_renorm.h"

namespace ctns{

// load operators from disk
template <typename Tm>
oper_dict<Tm> oper_load_qops(const comb<Tm>& icomb,
		             const comb_coord& p,
			     const std::string scratch,
			     const char type){
   oper_dict<Tm> qops;
   const auto& node = icomb.topo.get_node(p);
   if(type == 'c'){
      if(node.type != 3){
         auto fname0c = oper_fname(scratch, p, "cop"); // physical dofs
         oper_load(fname0c, qops);
      }else{
         auto pc = node.center;
         auto fname0c = oper_fname(scratch, pc, "rop"); // branching site
         oper_load(fname0c, qops);
      }
   }else if(type == 'r'){
      auto pr = node.right;
      auto fname0r = oper_fname(scratch, pr, "rop");
      oper_load(fname0r, qops);
   }else if(type == 'l'){
      auto pl = node.left;
      auto fname0l = oper_fname(scratch, pl, "lop");
      oper_load(fname0l, qops);
   }
   return qops;
}

// construct directly for boundary case {C,A,B,S,H}
template <typename Tm>
oper_dict<Tm> oper_init_local(const int kp,
		              const integral::two_body<Tm>& int2e,
		              const integral::one_body<Tm>& int1e){
   std::vector<int> krest;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      krest.push_back(k);
   }
   oper_dict<Tm> qops;
   oper_dot_C(kp, qops);
   oper_dot_A(kp, qops);
   oper_dot_B(kp, qops);
   oper_dot_P(kp, int2e, krest, qops);
   oper_dot_Q(kp, int2e, krest, qops);
   oper_dot_S(kp, int2e, int1e, krest, qops);
   oper_dot_H(kp, int2e, int1e, qops);
   return qops;
}

// construct directly for boundary case {C,A,B,S,H} [sites with type=0]
template <typename Tm>
void oper_init(const comb<Tm>& icomb,
	       const integral::two_body<Tm>& int2e,
	       const integral::one_body<Tm>& int1e,
	       const std::string scratch){
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      auto& node = icomb.topo.get_node(p);
      // local operators on physical sites
      if(node.type != 3){
         int kp = node.pindex;
         auto qops = oper_init_local(kp, int2e, int1e);
	 std::string fname = oper_fname(scratch, p, "cop");
         oper_save(fname, qops);
      }
      // right boundary (exclude the start point)
      if(node.type == 0 && p.first != 0){
         int kp = node.pindex;
         auto qops = oper_init_local(kp, int2e, int1e);
	 std::string fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   }
   // left boundary at the start
   auto p = std::make_pair(0,0);
   int kp = icomb.topo.nodes[0][0].pindex;
   auto qops = oper_init_local(kp, int2e, int1e);
   std::string fname = oper_fname(scratch, p, "lop");
   oper_save(fname, qops);
}

template <typename Tm>
void oper_env_right(const comb<Tm>& icomb, 
		    const integral::two_body<Tm>& int2e,
		    const integral::one_body<Tm>& int1e,
		    const std::string scratch){
   auto t0 = tools::get_time();
   std::cout << "ctns::oper_env_right" << std::endl;
   oper_init(icomb, int2e, int1e, scratch);
   // renormalization process
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      const auto& node = icomb.topo.get_node(p);
      if(node.type != 0 || p.first == 0){
         auto qops1 = oper_load_qops(icomb, p, scratch, 'c');
         auto qops2 = oper_load_qops(icomb, p, scratch, 'r');
	 const std::string superblock = "cr"; 
         auto qops = oper_renorm_ops(superblock, icomb, p, 
	        	 	     qops1, qops2, int2e, int1e);
	 auto fname = oper_fname(scratch, p, "rop");
         oper_save(fname, qops);
      }
   } // idx
   auto t1 = tools::get_time();
   std::cout << "timing for ctns::oper_env_right : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

template <typename Tm>
linalg::matrix<Tm> get_Hmat(const comb<Tm>& icomb, 
		            const integral::two_body<Tm>& int2e,
		            const integral::one_body<Tm>& int1e,
		            const double ecore,
		            const std::string scratch){
   std::cout << "\nctns::get_Hmat" << std::endl;
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, scratch);
   // load operators
   oper_dict<Tm> qops;
   auto p = std::make_pair(0,0); 
   auto fname = oper_fname(scratch, p, "rop");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows());
   
   // also deal with rwfuns, if necessary
   exit(1);

   return Hmat;
}

} // ctns

#endif
