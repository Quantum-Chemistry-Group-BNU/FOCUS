#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include <tuple>
#include "qtensor.h"

namespace ctns{

// container for operators
template <typename Tm>
using oper_map = std::map<int,qtensor2<Tm>>;
template <typename Tm>
using oper_dict = std::map<char,oper_map<Tm>>;

// pack two indices
const int kpack = 1000;
extern const int kpack;
// pack & unpack
inline int oper_pack(const int i, const int j){ 
   return i+j*kpack;
}
inline std::pair<int,int> oper_unpack(const int ij){
   return std::make_pair(ij%kpack,ij/kpack);
}

/*
// weight factor for AP/BQ pairs: ij - packed index
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
*/

} // ctns

#endif
