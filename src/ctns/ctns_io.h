#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../core/serialization.h"
#include "ctns_comb.h"

namespace ctns{ 

// for comb
template <typename Tm>
void rcanon_save(const comb<Tm>& icomb,
	         const std::string fname="rcanon.info"){
   std::cout << "\nctns::rcanon_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   //save << icomb.rsites << icomb.rwfuns;
   save << icomb.rwfuns;
   //save << icomb.rsites.at(std::make_pair(0,0));
}

template <typename Tm>
void rcanon_load(const comb<Tm>& icomb,
   	         const std::string fname="rcanon.info"){
   std::cout << "\nctns:rcanon_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   //load >> icomb.rsites >> icomb.rwfuns;
   load >> icomb.rwfuns;
}

} // ctns

#endif
