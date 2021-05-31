#ifndef CTNS_QDPT_H
#define CTNS_QDPT_H

#include "ctns_qsym.h"

namespace ctns{

// direct product table of qbond : V1*V2->V12
// qsym -> list of {(idx1,idx2,ioff)}
using qdpt = std::map<qsym,std::vector<std::tuple<int,int,int>>>;
using qproduct = std::pair<qbond,qdpt>;

inline qproduct qmerge(const qbond& qs1, const qbond& qs2){
   // init dpt
   qdpt dpt;
   for(int i1=0; i1<qs1.size(); i1++){
      auto q1 = qs1.get_sym(i1);
      for(int i2=0; i2<qs2.size(); i2++){
	 auto q2 = qs2.get_sym(i2);
	 dpt[q1+q2].push_back(std::make_tuple(i1,i2,0));
      }
   }
   // form qs12 & compute offset
   qbond qs12;
   for(auto& p : dpt){
      const auto& q12 = p.first;
      auto& p12 = p.second;
      int ioff = 0;
      for(int i12=0; i12<p12.size(); i12++){
         int i1 = std::get<0>(p12[i12]);
         int i2 = std::get<1>(p12[i12]);
         int d1 = qs1.get_dim(i1);
	 int d2 = qs2.get_dim(i2);
         p12[i12] = std::make_tuple(i1,i2,ioff);
	 ioff += d1*d2; 
      }
      qs12.dims.push_back(std::make_pair(q12,ioff));
   }
   return std::make_pair(qs12,dpt);
}

} // ctns

#endif
