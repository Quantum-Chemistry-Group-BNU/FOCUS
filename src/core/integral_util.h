#ifndef INTEGRAL_UTIL_H
#define INTEGRAL_UTIL_H

#include "integral.h"

namespace integral{

   template <typename Tm>
      void generate_N(one_body<Tm>& int1e){
         for(int i=0; i<int1e.sorb; i++){
            int1e.set(i, i, 1.0); 
         }
      }

   template <typename Tm>
      void generate_S2(two_body<Tm>& int2e,
            one_body<Tm>& int1e){

      }

} // integral

#endif
