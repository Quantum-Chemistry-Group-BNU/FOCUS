#ifndef CTNS_OPER_UTIL_H
#define CTNS_OPER_UTIL_H

#include <map>
#include <tuple>
#include "ctns_qtensor.h"

namespace ctns{

// container for operators	
template <typename Tm>
using oper_dict = std::map<char,std::map<int,qtensor2<Tm>>>;

// pack two indices
const int kpack = 500;
extern const int kpack;
// spincase = 0/1 : same/different spin
// val = even: same spin
// val = odd : different spin
inline int oper_pack(const int spincase, const int i, const int j){
   return spincase + (i+j*kpack)*2;
}
inline std::tuple<int,int,int> oper_unpack(const int ij){
   return std::make_tuple(ij%2, (ij/2)%kpack, (ij/2)/kpack);
}
inline std::pair<int,int> oper_unpack2(const int ij){
   int spincase = ij%2;
   int kp = (ij/2)%kpack;
   int kq = (ij/2)/kpack;
   return std::make_pair(2*kp, 2*kq+spincase);
}

// weight factor for AP/BQ pairs
inline double wfacAP(const int ij){
   int spincase = ij%2;
   int kp = (ij/2)%kpack;
   int kq = (ij/2)/kpack;
   return (spincase==1 && kp==kq)? 0.5 : 1.0; 
}
inline double wfacBQ(const int ij){
   int spincase = ij%2;
   int kp = (ij/2)%kpack;
   int kq = (ij/2)/kpack;
   return (kp==kq)? 0.5 : 1.0; 
}

} // ctns

#endif
