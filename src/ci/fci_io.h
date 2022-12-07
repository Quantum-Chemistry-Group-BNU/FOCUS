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
   ofs.close();
   // ZL@20221207 binary format for easier loading in python 
   std::ofstream ofs2(fname+".bin", std::ios::binary);
   fock::onspace_compact space_compact(space);
   space_compact.save(ofs2);
   int nroot = es.size();
   ofs2.write((char*)(&nroot), sizeof(nroot));
   ofs2.write((char*)(es.data()), sizeof(double)*nroot);
   for(int i=0; i<nroot; i++){
      ofs2.write((char*)(vs[i].data()), sizeof(Tm)*vs[i].size());
   }
   ofs2.close();
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
   ifs.close();
}

} // fci

#endif
