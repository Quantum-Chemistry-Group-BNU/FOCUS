#ifndef CTNS_TOSU2_QBOND3_H
#define CTNS_TOSU2_QBOND3_H

#include <set>
#include "../../qtensor/qnum_qbond.h"

namespace ctns{

   using qsym3 = std::tuple<short,short,short>; // (N,TS,TM)
   using qbond3 = std::vector<std::pair<qsym3,int>>; // qsym,dim

   inline std::pair<qbond,std::map<qsym,int>> qbond3_to_qbond(const qbond3& qs3){
      qbond qs;
      std::map<qsym,int> symdict;
      int idx = 0;
      for(int i=0; i<qs3.size(); i++){
         const auto& sym3 = qs3[i].first;
         qsym sym({3,std::get<0>(sym3),std::get<1>(sym3)});
         if(symdict.find(sym) == symdict.end()){
            symdict[sym] = idx;
            idx += 1;
            qs.dims.push_back(std::make_pair(sym,qs3[i].second));
         }
      } // i
      return std::make_pair(qs,symdict);
   }

   inline std::vector<int> get_offset(const qbond3& qs){
      std::vector<int> offset;
      int ioff = 0;
      for(int i=0; i<qs.size(); i++){
         offset.push_back(ioff);
         ioff += qs[i].second; 
      }
      return offset;
   }

   inline int get_dimAll(const qbond3& qs){
      int dim = 0;
      for(const auto& p : qs) dim += p.second;
      return dim;
   }

   inline void display_qbond3(const qbond3& qs, 
         const std::string name,
         const bool debug=true){
      std::cout << " qbond3: " << name
         << " nsym=" << qs.size()
         << " dimAll=" << get_dimAll(qs)
         << std::endl;
      if(debug){
         for(int i=0; i<qs.size(); i++){
            auto sym = qs[i].first;
            auto dim = qs[i].second;
            std::cout << " (" << std::get<0>(sym) << "," 
               << std::get<1>(sym) << "," 
               << std::get<2>(sym) << ")" 
               << ":" << dim;
         }
         std::cout << std::endl;
      }
   }

   inline qbond3 get_qbond3_vac(const int ts=0){
      qbond3 qvac;
      for(int tm=-ts; tm<=ts; tm+=2){
         qsym3 sym = std::make_tuple(ts,ts,tm);
         qvac.push_back(std::make_pair(sym,1));
      }
      return qvac; 
   }

   inline qbond3 get_qbond_phys(){
      qbond3 qphys({{std::make_tuple(0,0,0),1},   // 0
            {std::make_tuple(2,0,0),1},   // 2
            {std::make_tuple(1,1,1),1},   // a
            {std::make_tuple(1,1,-1),1}}); // b
      return qphys;
   }

} // ctns

#endif
