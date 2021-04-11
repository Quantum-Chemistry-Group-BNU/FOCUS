#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"

namespace ctns{ 

const bool debug_oper_io = true;
extern const bool debug_oper_io;

// for qopers
std::string oper_fname(const std::string scratch, 
  	  	       const comb_coord& p,
		       const std::string optype){
   std::string fname = scratch + "/" + optype + "("
	             + std::to_string(p.first) + ","
	             + std::to_string(p.second) + ")";
   return fname;
}

// add individual operator in future
template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops){
   if(debug_oper_io){
      std::cout << "ctns::oper_save fname=" << fname 
 	        << " size=" << qops.size() << std::endl;
   }
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops){
   if(debug_oper_io){
      std::cout << "ctns::oper_load fname=" << fname 
	        << std::endl;
   }
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}

} // ctns

#endif
