#ifndef SPINCOUPLING_H
#define SPINCOUPLING_H

#include "../core/spin.h"

namespace ctns{

   // three spin coupling (LC)R, L(CR)
   enum spincoupling3 {LCcouple, CRcouple};
   // four spin coupling (LC1)(C2R), ((LC1)C2)R, L(C1(C2R))
   enum spincoupling4 {LC1andC2Rcouple, LC1andC2couple, C1andC2Rcouple};

} // ctns

#endif
