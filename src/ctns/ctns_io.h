#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../core/serialization.h"
#include "ctns_comb.h"
#include "ctns_oper_util.h"

namespace ctns{ 

// for comb
template <typename Tm>
void rcanon_save(const comb<Tm>& icomb,
	         const std::string fname="rcanon.info"){
   std::cout << "\nctns::rcanon_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << icomb.rsites << icomb.rwfuns;
}

template <typename Tm>
void rcanon_load(comb<Tm>& icomb, // no const!
   	         const std::string fname="rcanon.info"){
   std::cout << "\nctns:rcanon_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> icomb.rsites >> icomb.rwfuns;
}

// for qopers
std::string oper_fname(const std::string scratch, 
  	  	       const comb_coord& p,
		       const std::string optype){
   std::string fname = scratch + "/" + optype + "("
	             + std::to_string(p.first) + ","
	             + std::to_string(p.second) + ")";
   return fname;
}

template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops){
   const bool debug = false;
   if(debug){
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
   const bool debug = false;
   if(debug){
      std::cout << "ctns::oper_load fname=" << fname 
	        << std::endl;
   }
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
}

} // ctns

#endif
