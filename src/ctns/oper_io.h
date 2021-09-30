#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"
#include "ctns_comb.h"

namespace ctns{ 

const bool debug_oper_io = false;
extern const bool debug_oper_io;

inline std::string oper_fname(const std::string scratch, 
  	  	       	      const comb_coord& p,
		       	      const std::string kind){
   return scratch + "/" + kind + "_"
        + std::to_string(p.first) + "_"
        + std::to_string(p.second) + ".op";
}

template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops){
   if(debug_oper_io) std::cout << " ctns::oper_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops){
   if(debug_oper_io) std::cout << " ctns::oper_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}

//
// load operators from disk for site p
//
//       cop
//        |
// lop ---*--- rop
//	  p
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

} // ctns

#endif
