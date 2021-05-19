#ifndef OPER_UTIL_H
#define OPER_UTIL_H

#include "oper_dict.h"
#include "oper_io.h"
#include "oper_dot.h"
#include "oper_renorm.h"

namespace ctns{

// load operators from disk for site p
//
//       cop
//        |
// lop ---*--- rop
//
template <typename Km>
void oper_load_qops(const comb<Km>& icomb,
     		    const comb_coord& p,
     		    const std::string scratch,
		    const std::string block,
		    oper_dict<typename Km::dtype>& qops){
   const auto& node = icomb.topo.get_node(p);
   if(block == "c"){
      if(node.type != 3){
         auto fname0c = oper_fname(scratch, p, "c"); // physical dofs
         oper_load(fname0c, qops);
      }else{
         auto pc = node.center;
         auto fname0c = oper_fname(scratch, pc, "r"); // branching site
         oper_load(fname0c, qops);
      }
   }else if(block == "r"){
      auto pr = node.right;
      auto fname0r = oper_fname(scratch, pr, "r");
      oper_load(fname0r, qops);
   }else if(block == "l"){
      auto pl = node.left;
      auto fname0l = oper_fname(scratch, pl, "l");
      oper_load(fname0l, qops);
   }
}

// init local operators
template <typename Tm>
void oper_init_dot(const int isym,
		   const bool ifkr,
		   const int kp,
		   const integral::two_body<Tm>& int2e,
		   const integral::one_body<Tm>& int1e,
		   oper_dict<Tm>& qops){
   std::vector<int> krest;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      krest.push_back(k);
   }
   oper_dot_opC(isym, ifkr, kp, qops);
   oper_dot_opA(isym, ifkr, kp, qops);
   oper_dot_opB(isym, ifkr, kp, qops);
   oper_dot_opP(isym, ifkr, kp, int2e, krest, qops);
   oper_dot_opQ(isym, ifkr, kp, int2e, krest, qops);
   oper_dot_opS(isym, ifkr, kp, int2e, int1e, krest, qops);
   oper_dot_opH(isym, ifkr, kp, int2e, int1e, qops);
}

// initialization of operators for 
// (1) dot operators [c] 
// (2) boundary operators [l/r]
template <typename Km>
void oper_init(const comb<Km>& icomb,
	       const integral::two_body<typename Km::dtype>& int2e,
	       const integral::one_body<typename Km::dtype>& int1e,
	       const std::string scratch){ 
   const bool ifkr = kind::is_kramers<Km>();
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      auto& node = icomb.topo.get_node(p);
      // cop: local operators on physical sites
      if(node.type != 3){
         int kp = node.pindex;
         oper_dict<typename Km::dtype> qops;
	 oper_init_dot(Km::isym, ifkr, kp, int2e, int1e, qops);	
	 std::string fname = oper_fname(scratch, p, "c");
         oper_save(fname, qops);
      }
      // rop: right boundary (exclude the start point)
      if(node.type == 0 && p.first != 0){
	 int kp = node.pindex;
         oper_dict<typename Km::dtype> qops;
	 oper_init_dot(Km::isym, ifkr, kp, int2e, int1e, qops);
	 std::string fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
      }
   }
   // left boundary at the start (0,0)
   auto p = std::make_pair(0,0);
   int kp = icomb.topo.get_node(p).pindex;
   oper_dict<typename Km::dtype> qops;
   oper_init_dot(Km::isym, ifkr, kp, int2e, int1e, qops);
   std::string fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
}

// build right environment operators
template <typename Km>
void oper_env_right(const comb<Km>& icomb, 
		    const integral::two_body<typename Km::dtype>& int2e,
		    const integral::one_body<typename Km::dtype>& int1e,
		    const std::string scratch){
   auto t0 = tools::get_time();
   std::cout << "ctns::oper_env_right" << std::endl;
   // construct for dot operators [cop] & boundary operators [lop/rop]
   oper_init(icomb, int2e, int1e, scratch);
   // successive renormalization process
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      const auto& node = icomb.topo.get_node(p);
      if(node.type != 0 || p.first == 0){
	 oper_dict<typename Km::dtype> qops1, qops2, qops; 
         // load operators from disk    
         oper_load_qops(icomb, p, scratch, "c", qops1);
         oper_load_qops(icomb, p, scratch, "r", qops2);
	 if(debug_oper_dict){
	    oper_display(qops1, "c", 1);
	    oper_display(qops2, "r", 1);
	 }
         // perform renormalization for superblock {|cr>}
	 std::string superblock = "cr"; 
         oper_renorm_opAll(superblock, icomb, p, int2e, int1e, qops1, qops2, qops);
	 auto fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
      }
   } // idx
   auto t1 = tools::get_time();
   std::cout << "timing for ctns::oper_env_right : " << std::setprecision(2) 
             << tools::get_duration(t1-t0) << " s" << std::endl;
}

// Hij = <CTNS[i]|H|CTNS[j]>
template <typename Km>
linalg::matrix<typename Km::dtype> get_Hmat(const comb<Km>& icomb, 
		            		    const integral::two_body<typename Km::dtype>& int2e,
		            		    const integral::one_body<typename Km::dtype>& int1e,
		            		    const double ecore,
		            		    const std::string scratch){
   std::cout << "\nctns::get_Hmat" << std::endl;
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, scratch);

   // load operators from file
   oper_dict<typename Km::dtype> qops;
   auto p = std::make_pair(0,0); 
   auto fname = oper_fname(scratch, p, "r");
   oper_load(fname, qops);
   auto Hmat = qops['H'][0].to_matrix();
   Hmat += ecore*linalg::identity_matrix<typename Km::dtype>(Hmat.rows());
   // deal with rwfuns(istate,ibas):
   // Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T) 
   auto wfmat = icomb.rwfuns.to_matrix();
   auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
   Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
   return Hmat;
}

} // ctns

#endif
