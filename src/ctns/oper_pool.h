#ifndef OPER_POOL_H
#define OPER_POOL_H

#include "oper_pool_raw.h"
#include "oper_pool_safe.h"

namespace ctns{
 
   // Change the definition of oper_pool here
   template <typename Tm>
      using oper_pool = oper_pool_raw<Tm>;
      //using oper_pool = oper_pool_safe<Tm>;

} // ctns

#endif
