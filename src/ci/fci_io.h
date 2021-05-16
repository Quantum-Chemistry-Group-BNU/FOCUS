#ifndef FCI_IO_H
#define FCI_IO_H

#include "../core/serialization.h"

namespace fci{

// io: save/load onspace & ci vectors
template <typename Tm>
void ci_save(const fock::onspace& space,
	     const std::vector<double>& es,
	     const std::vector<std::vector<Tm>>& vs,
	     const std::string fname="ci.info"){
   std::cout << "\nfci::ci_save fname=" << fname << std::endl;
   std::ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << space << es << vs;
}

template <typename Tm>
void ci_load(fock::onspace& space,
	     std::vector<double>& es,
	     std::vector<std::vector<Tm>>& vs,
	     const std::string fname="ci.info"){
   std::cout << "\nfci::ci_load fname=" << fname << std::endl;
   std::ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> space >> es >> vs;
}

} // fci

#endif
