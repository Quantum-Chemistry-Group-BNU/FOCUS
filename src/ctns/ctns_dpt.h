#ifndef CTNS_DPT_H
#define CTNS_DPT_H

#include "ctns_qsym.h"

namespace ctns{

// direct product table of qbond : V1*V2->V12
// map from (idx1,idx2)->(d1,d2,ioff)
using qdpt = std::map<qsym,std::map<std::pair<int,int>,std::tuple<int,int,int>>>;

std::pair<qbond,qdpt> merge(const qbond& qs1, const qbond& qs2){
   // init dpt
   qdpt dpt;
   for(int i1=0; i1<qs1.size(); i1++){
      auto q1 = qs1.get_sym(i1);
      auto d1 = qs1.get_dim(i1);
      for(int i2=0; i2<qs2.size(); i2++){
	 auto q2 = qs2.get_sym(i2);
	 auto d2 = qs2.get_dim(i2);
	 dpt[q1+q2][std::make_pair(i1,i2)] = std::make_tuple(d1,d2,0);
      }
   }
   // form qs12 & compute offset
   qbond qs12;
   for(const auto& p : dpt){
      const auto& q12 = p.first;
      int ioff = 0;
      for(const auto& p12 : p.second){
         int d1 = std::get<0>(p12.second);
         int d2 = std::get<1>(p12.second);
         dpt[q12][p12.first] = std::make_tuple(d1,d2,ioff);
	 ioff += d1*d2; 
      }
      qs12.dims.push_back(std::make_pair(q12,ioff));
   }
   return std::make_pair(qs12,dpt);
}

} // ctns

#endif
