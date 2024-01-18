#ifndef CTNS_TOSU2_H
#define CTNS_TOSU2_H

#include "../init_phys.h"
#include "ctns_tosu2_qbond3.h"
#include "ctns_tosu2_wmat.h"
#include "ctns_tosu2_site.h"
#include "ctns_tosu2_dm.h"

namespace ctns{

   // convert to SU2 symmetry via sweep projection
   template <typename Tm>
      void rcanon_tosu2(const comb<qkind::qNSz,Tm>& icomb_NSz,
            comb<qkind::qNS,Tm>& icomb,
            const int twos){
         std::cout << "\nctns::rcanon_tosu2 twos=" << twos << std::endl;
         auto t0 = tools::get_time();

         // Algorithm:

         // initial Wmatrix

         // sweep projection: start from the last site
         int nsite = icomb_NSz.get_nphysical();
         for(int i=nsite-1; i>=0; i--){

            // form MixedSite

            // decimation by diagonlizing the quasi-density matrix
    
            // update information: W & mps site

         }

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_tosu2", t0, t1);
      }

} // ctns

#endif
