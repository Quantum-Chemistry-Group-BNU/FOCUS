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
const int kpack = 1000;
extern const int kpack;

inline int oper_pack(const int i, const int j){ 
   return i+j*kpack;
}

inline std::pair<int,int> oper_unpack(const int ij){
   return std::make_pair(ij%kpack,ij/kpack);
}

} // ctns

#endif
