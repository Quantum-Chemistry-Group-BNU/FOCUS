#ifndef VMC_IO_H
#define VMC_IO_H

#include "../core/serialization.h"
#include "ansatz.h"

namespace ctns{ 

   // for comb
   template <typename Km>
      void wf_save(const BaseAnsatz& wavefun,
            const std::string fname){
         std::cout << "\nvmc::wf_save fname=" << fname << std::endl;
         std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         save << wavefun;
         ofs.close();
      }

   template <typename Km>
      void wf_load(BaseAnsatz<Km>& icomb, // no const!
            const std::string fname){
         std::cout << "\nvmc:vmc_load fname=" << fname << std::endl;
         std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         load >> wavefun;
         ifs.close();
      }

} // vmc

#endif
