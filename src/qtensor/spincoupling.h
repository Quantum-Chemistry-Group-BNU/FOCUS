#ifndef SPINCOUPLING_H
#define SPINCOUPLING_H

namespace ctns{

   // three spin coupling (LC)R, L(CR)
   enum spincoupling3 {LCcouple, CRcouple};
   // four spin coupling (LC1)(C2R), ((LC1)C2)R, L(C1(C2R))
   enum spincoupling4 {LC1andC2Rcouple, LC1andC2couple, C1andC2Rcouple};

   inline bool spin_triangle(const int s1, const int s2, const int s3){
      return std::abs(s1-s2) <= s3 && s3 <= s1+s2;
   }

} // ctns

#endif
