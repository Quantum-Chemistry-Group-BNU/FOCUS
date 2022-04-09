#ifndef OPER_IO_H
#define OPER_IO_H

#include "../core/serialization.h"
#include "oper_dict.h"
#include "ctns_comb.h"

namespace ctns{ 

const bool debug_oper_io = true;
extern const bool debug_oper_io;

inline std::string oper_fname(const std::string scratch, 
  	  	       	      const comb_coord& p,
		       	      const std::string kind){
   return scratch + "/" + kind + "("
        + std::to_string(p.first) + ","
        + std::to_string(p.second) + ").op";
}

template <typename Tm>
void oper_save(const std::string fname, 
	       const oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug) std::cout << "ctns::oper_save fname=" << fname;
   auto t0 = tools::get_time();
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << qops;
   auto t1 = tools::get_time();
   ofs.write(reinterpret_cast<const char*>(qops._data), qops._size*sizeof(Tm));
   ofs.close();
   auto t2 = tools::get_time();
   if(debug_oper_io and debug){
      std::cout << " T(info/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tools::get_duration(t2-t0) 
                << std::endl;
   }
}

template <typename Tm>
void oper_load(const std::string fname, 
	       oper_dict<Tm>& qops,
	       const bool debug){
   if(debug_oper_io and debug) std::cout << "ctns::oper_load fname=" << fname;
   auto t0 = tools::get_time();
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> qops;
   auto t1 = tools::get_time();
   qops._setup_opdict();
   qops._data = new Tm[qops._size];
   auto t2 = tools::get_time();
   ifs.read(reinterpret_cast<char*>(qops._data), qops._size*sizeof(Tm));
   ifs.close();
   qops._setup_data(qops._data);
   auto t3 = tools::get_time();
   if(debug_oper_io and debug){
      std::cout << " T(info/setup/data/tot)=" 
                << tools::get_duration(t1-t0) << "," 
                << tools::get_duration(t2-t1) << ","
                << tools::get_duration(t3-t2) << "," 
                << tools::get_duration(t3-t0) 
                << std::endl;
   }
}

} // ctns

#endif
