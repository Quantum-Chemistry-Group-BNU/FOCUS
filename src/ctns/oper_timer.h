#ifndef OPER_TIMER_H
#define OPER_TIMER_H

#include "../core/tools.h"

namespace ctns{

struct oper_timing{
   void analysis(){
      std::cout << "----- oper_timing -----" << std::endl;
      std::cout << " opC: n=" << nC << " t=" << tC << " S" << " tav=" << tC/nC << " S" << std::endl;
      std::cout << " opA: n=" << nA << " t=" << tA << " S" << " tav=" << tA/nA << " S" << std::endl;
      std::cout << " opB: n=" << nB << " t=" << tB << " S" << " tav=" << tB/nB << " S" << std::endl;
      std::cout << " opH: n=" << nH << " t=" << tH << " S" << " tav=" << tH/nH << " S" << std::endl;
      std::cout << " opS: n=" << nS << " t=" << tS << " S" << " tav=" << tS/nS << " S" << std::endl;
      std::cout << " opP: n=" << nP << " t=" << tP << " S" << " tav=" << tP/nP << " S" << std::endl;
      std::cout << " opQ: n=" << nQ << " t=" << tQ << " S" << " tav=" << tQ/nQ << " S" << std::endl;
   }
   void clear(){
      nC=0; nA=0; nB=0; nH=0; nS=0; nP=0; nQ=0;
      tC=0.0; tA=0.0; tB=0.0; tH=0.0; tS=0.0; tP=0.0; tQ=0.0;
   }
public:
   int nC=0, nA=0, nB=0, nH=0, nS=0, nP=0, nQ=0;
   double tC=0.0, tA=0.0, tB=0.0, tH=0.0, tS=0.0, tP=0.0, tQ=0.0;
};

extern oper_timing oper_timer;

} // ctns

#endif
