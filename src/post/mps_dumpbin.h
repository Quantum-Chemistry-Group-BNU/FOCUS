#ifndef MPS_DUMPBIN_H
#define MPS_DUMPBIN_H

#include "mps.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void mps_dumpbin(const input::schedule& schd){
         std::cout << "\nctns::mps_dumpbin" << std::endl;
         topology topo;
         topo.read(schd.post.topology_file);
         //topo.print();
         int nket = schd.post.ket.size();
         for(int j=0; j<nket; j++){
            std::cout << "\n### jket=" << j << " ###" << std::endl;
            mps<Qm,Tm> kmps;
            auto kmps_file = schd.scratch+"/rcanon_isweep"+std::to_string(schd.post.ket[j]); 
            kmps.nphysical = topo.nphysical;
            kmps.image2 = topo.image2;
            kmps.load(kmps_file);
            // convert to binary format
            kmps.dumpbin(kmps_file);
         }
      }

} // ctns

#endif
