#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"

namespace ctns{ 

// for qopers
inline std::string oper_fname(const std::string scratch, 
  	  	       	      const comb_coord& p,
		       	      const std::string kind){
   return scratch + "/" + kind + "_"
        + std::to_string(p.first) + "_"
        + std::to_string(p.second) + ".op";
}

// add individual operator in future
template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops){
   if(debug_oper_io) std::cout << "ctns::oper_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops){
   if(debug_oper_io) std::cout << "ctns::oper_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}

} // ctns

#endif
