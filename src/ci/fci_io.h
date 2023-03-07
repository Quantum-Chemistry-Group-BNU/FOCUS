#ifndef FCI_IO_H
#define FCI_IO_H

#include "../core/serialization.h"

namespace fci{

   const bool ifsavebin = false;
   extern const bool ifsavebin;

   // io: save/load onspace & ci vectors
   template <typename Tm>
      void ci_save(const fock::onspace& space,
            const std::vector<double>& es,
            const linalg::matrix<Tm>& vs,
            const std::string fname="ci.info"){
         std::cout << "\nfci::ci_save fname=" << fname << std::endl;

         std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         save << space << es << vs;
         ofs.close();

         // ZL@20221207 binary format for easier loading in python 
         if(ifsavebin){
            std::ofstream ofs2(fname+".bin", std::ios::binary);
            fock::onspace_compact space_compact(space);
            space_compact.dump(ofs2);
            int nroots = es.size();
            int dim = vs.rows();
            ofs2.write((char*)(&nroots), sizeof(nroots));
            ofs2.write((char*)(es.data()), sizeof(double)*nroots);
            ofs2.write((char*)(vs.data()), sizeof(Tm)*dim*nroots);
            ofs2.close();
         }
      }

   template <typename Tm>
      void ci_load(fock::onspace& space,
            std::vector<double>& es,
            linalg::matrix<Tm>& vs,
            const std::string fname="ci.info"){
         std::cout << "\nfci::ci_load fname=" << fname << std::endl;
         std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         load >> space >> es >> vs;
         ifs.close();
      }

} // fci

#endif
