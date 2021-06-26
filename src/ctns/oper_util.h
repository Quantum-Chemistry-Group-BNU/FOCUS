#ifndef OPER_UTIL_H
#define OPER_UTIL_H

#include "oper_dict.h"
#include "oper_io.h"
#include "oper_dot.h"
#include "oper_renorm.h"
#ifndef SERIAL
#include <boost/mpi.hpp>
#endif

namespace ctns{

//
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
		    const std::string kind,
		    oper_dict<typename Km::dtype>& qops){
   const auto& node = icomb.topo.get_node(p);
   if(kind == "c"){
      if(node.type != 3){
         auto fname0c = oper_fname(scratch, p, "c"); // physical dofs
         oper_load(fname0c, qops);
      }else{
         auto pc = node.center;
         auto fname0c = oper_fname(scratch, pc, "r"); // branching site
         oper_load(fname0c, qops);
      }
   }else if(kind == "r"){
      auto pr = node.right;
      auto fname0r = oper_fname(scratch, pr, "r");
      oper_load(fname0r, qops);
   }else if(kind == "l"){
      auto pl = node.left;
      auto fname0l = oper_fname(scratch, pl, "l");
      oper_load(fname0l, qops);
   }
}

// init local operators
template <typename Km>
void oper_init_dot(const comb<Km>& icomb,
		   const int kp,
		   const integral::two_body<typename Km::dtype>& int2e,
		   const integral::one_body<typename Km::dtype>& int1e,
		   oper_dict<typename Km::dtype>& qops){
   const int isym = Km::isym;
   const bool ifkr = kind::is_kramers<Km>();
   // rest of spatial orbital indices
   std::vector<int> krest;
   for(int k=0; k<int1e.sorb/2; k++){
      if(k == kp) continue;
      krest.push_back(k);
   }
   qops.cindex.push_back(2*kp);
   if(not ifkr) qops.cindex.push_back(2*kp+1);
   // compute
   oper_dot_opC(isym, ifkr, kp, qops);
   oper_dot_opA(isym, ifkr, kp, qops);
   oper_dot_opB(isym, ifkr, kp, qops);
   oper_dot_opP(isym, ifkr, kp, int2e, krest, qops);
   oper_dot_opQ(isym, ifkr, kp, int2e, krest, qops);
   oper_dot_opS(isym, ifkr, kp, int2e, int1e, krest, qops);
   oper_dot_opH(isym, ifkr, kp, int2e, int1e, qops);
   // scale full {Sp,H} on dot to avoid repetition in parallelization
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
   if(size > 1){
      const typename Km::dtype scale = 1.0/size;
      qops('H')[0] = qops('H')[0]*scale;
      for(auto& p : qops('S')){
         p.second = p.second*scale;
      }
   }
#endif
}

// initialization of operators for 
// (1) dot operators [c] 
// (2) boundary operators [l/r]
template <typename Km>
void oper_init(const comb<Km>& icomb,
	       const integral::two_body<typename Km::dtype>& int2e,
	       const integral::one_body<typename Km::dtype>& int1e,
	       const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif  
   double t_comp = 0.0, t_save = 0.0; 
   auto t0 = tools::get_time();
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      auto& node = icomb.topo.get_node(p);
      // cop: local operators on physical sites
      if(node.type != 3){
         int kp = node.pindex;
         auto ta = tools::get_time();
         oper_dict<typename Km::dtype> qops;
	 oper_init_dot(icomb, kp, int2e, int1e, qops);	
         auto tb = tools::get_time();
	 std::string fname = oper_fname(scratch, p, "c");
         oper_save(fname, qops);
         auto tc = tools::get_time();
         t_comp += tools::get_duration(tb-ta);
         t_save += tools::get_duration(tc-tb);
      }
      // rop: right boundary (exclude the start point)
      if(node.type == 0 && p.first != 0){
	 int kp = node.pindex;
         auto ta = tools::get_time();
         oper_dict<typename Km::dtype> qops;
	 oper_init_dot(icomb, kp, int2e, int1e, qops);
         auto tb = tools::get_time();
	 std::string fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
         auto tc = tools::get_time();
         t_comp += tools::get_duration(tb-ta);
         t_save += tools::get_duration(tc-tb);
      }
   }
   // left boundary at the start (0,0)
   auto p = std::make_pair(0,0);
   int kp = icomb.topo.get_node(p).pindex;
   auto ta = tools::get_time();
   oper_dict<typename Km::dtype> qops;
   oper_init_dot(icomb, kp, int2e, int1e, qops);
   auto tb = tools::get_time();
   std::string fname = oper_fname(scratch, p, "l");
   oper_save(fname, qops);
   auto tc = tools::get_time();
   t_comp += tools::get_duration(tb-ta);
   t_save += tools::get_duration(tc-tb);
   auto t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::oper_init", t0, t1);
      std::cout << "detailed t[comp/save]= "
	        << t_comp << ", " << t_save
                << " t[total]=" << (t_comp + t_save)
                << std::endl;
   }
}

// build right environment operators
template <typename Km>
void oper_env_right(const comb<Km>& icomb, 
		    const integral::two_body<typename Km::dtype>& int2e,
		    const integral::one_body<typename Km::dtype>& int1e,
		    const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "ctns::oper_env_right" << std::endl;
   double t_init = 0.0, t_load = 0.0, t_comp = 0.0, t_save = 0.0;
   // construct for dot operators [cop] & boundary operators [lop/rop]
   auto t0 = tools::get_time();
   oper_init(icomb, int2e, int1e, scratch);
   auto ta = tools::get_time();
   t_init = tools::get_duration(ta-t0);
   // successive renormalization process
   for(int idx=0; idx<icomb.topo.rcoord.size(); idx++){
      auto p = icomb.topo.rcoord[idx];
      const auto& node = icomb.topo.get_node(p);
      if(node.type != 0 || p.first == 0){
	 oper_dict<typename Km::dtype> qops1, qops2, qops; 
         // load operators from disk    
         auto tb = tools::get_time();
         oper_load_qops(icomb, p, scratch, "c", qops1);
         oper_load_qops(icomb, p, scratch, "r", qops2);
         auto tc = tools::get_time();
         t_load += tools::get_duration(tc-tb); 
	 if(debug_oper_dict){
	    qops1.print("qops_c", 1);
	    qops2.print("qops_r", 1);
	 }
         // perform renormalization for superblock {|cr>}
	 std::string superblock = "cr"; 
         oper_renorm_opAll(superblock, icomb, p, int2e, int1e, qops1, qops2, qops);
         auto td = tools::get_time();
         t_comp += tools::get_duration(td-tc);
	 // save operators to disk
	 std::string fname = oper_fname(scratch, p, "r");
         oper_save(fname, qops);
         auto te = tools::get_time();
	 t_save += tools::get_duration(te-td);
         if(rank == 0) qops.print(fname);
      }
   } // idx
   auto t1 = tools::get_time();
   if(rank == 0){
      tools::timing("ctns::oper_env_right", t0, t1);
      std::cout << "detailed t[init/load/comp/save]= "
                << t_init << ", " << t_load << ", " << t_comp << ", " << t_save
                << " t[total]=" << (t_init + t_load + t_comp + t_save)
                << std::endl;
   }
}

// Hij = <CTNS[i]|H|CTNS[j]>
template <typename Km>
linalg::matrix<typename Km::dtype> get_Hmat(const comb<Km>& icomb, 
		            		    const integral::two_body<typename Km::dtype>& int2e,
		            		    const integral::one_body<typename Km::dtype>& int1e,
		            		    const double ecore,
		            		    const std::string scratch){
   int size = 1, rank = 0;
#ifndef SERIAL
   size = icomb.world.size();
   rank = icomb.world.rank();
#endif   
   if(rank == 0) std::cout << "\nctns::get_Hmat" << std::endl;
   // build operators for environement
   oper_env_right(icomb, int2e, int1e, scratch);
   // load operators from file
   using Tm = typename Km::dtype;
   oper_dict<Tm> qops;
   auto p = std::make_pair(0,0); 
   auto fname = oper_fname(scratch, p, "r");
   oper_load(fname, qops);
   auto Hmat = qops('H')[0].to_matrix();
   if(rank == 0) Hmat += ecore*linalg::identity_matrix<Tm>(Hmat.rows()); // avoid repetition
   // deal with rwfuns(istate,ibas): Hij = w*[i,a] H[a,b] w[j,b] = (w^* H w^T) 
   auto wfmat = icomb.rwfuns.to_matrix();
   auto tmp = linalg::xgemm("N","T",Hmat,wfmat);
   Hmat = linalg::xgemm("N","N",wfmat.conj(),tmp);
#ifndef SERIAL
   // reduction of partial H formed on each processor
   if(size > 1){
      linalg::matrix<Tm> Hmat2(Hmat.rows(),Hmat.cols());
      boost::mpi::reduce(icomb.world, Hmat, Hmat2, std::plus<linalg::matrix<Tm>>(), 0);
      Hmat = Hmat2;
   }
#endif  
   return Hmat;
}

} // ctns

#endif
