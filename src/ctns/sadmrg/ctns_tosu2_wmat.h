#ifndef CTNS_TOSU2_WMAT_H
#define CTNS_TOSU2_WMAT_H

#include "../init_phys.h"
#include "ctns_tosu2_qbond3.h"

namespace ctns{

   // W[alpha,beta]
   template <typename Tm>
      struct Wmatrix{

         public:
            qbond qrow;
            qbond3 qcol;
            std::vector<Tm> data;
      };

   template <typename Tm>
      void initW0vac(const int ts=0, const int tm=0){
         qbond qrow = get_qbond_vac(2); 
         qbond3 qcol = get_qbond3_vac(ts); 

      };

} // ctns

#endif
