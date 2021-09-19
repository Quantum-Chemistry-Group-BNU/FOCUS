#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../core/serialization.h"
#include "ctns_comb.h"

namespace ctns{ 

// for comb
template <typename Km>
void rcanon_save(const comb<Km>& icomb,
	         const std::string fname="rcanon.info"){
   std::cout << "\nctns::rcanon_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   // save rsites & rwfuns
   for(int idx=0; idx<icomb.topo.ntotal; idx++){
      save << icomb.rsites[idx];
   }
   save << icomb.rwfuns;
}

template <typename Km>
void rcanon_load(comb<Km>& icomb, // no const!
   	         const std::string fname="rcanon.info"){
   std::cout << "\nctns:rcanon_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   // load rsites & rwfuns
   icomb.rsites.resize(icomb.topo.ntotal);
   for(int idx=0; idx<icomb.topo.ntotal; idx++){
      // this save calls to copy constructor for vector<st3>
      load >> icomb.rsites[idx]; 
   }
   load >> icomb.rwfuns;
}

} // ctns

#endif
