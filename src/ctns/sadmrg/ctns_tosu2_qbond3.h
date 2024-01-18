#ifndef CTNS_TOSU2_QBOND3_H
#define CTNS_TOSU2_QBOND3_H

namespace ctns{

   using qsym3 = std::tuple<short,short,short>; // (N,TS,TM)
   using qbond3 = std::vector<std::pair<qsym3,int>>; // qsym,dim

   inline qbond3 get_qbond3_vac(const int ts=0){
      qbond3 qvac;
      for(int tm=-ts; tm<=ts; tm+=2){
         qsym3 sym = std::make_tuple(ts,ts,tm);
         qvac.push_back(std::make_pair(sym,1));
      }
      return qvac; 
   }

} // ctns

#endif
