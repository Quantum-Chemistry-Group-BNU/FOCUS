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

// --- packing ---
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

// --- weight factor ---
// ifkr = false
inline double wfac(const int ij){
   int i = ij%kpack;
   int j = ij/kpack;
   return (i==j)? 0.5 : 1.0; 
}
// ifkr = true
inline double wfacAP(const int ij){
   int i = ij%kpack, ki = i/2, spin_i = i%2;
   int j = ij/kpack, kj = j/2, spin_j = j%2;
   if(spin_i == spin_j){
      return 1.0;
   }else{
      return (ki==kj)? 0.5 : 1.0; // avoid duplication for A[p\bar{p}]
   }
}
inline double wfacBQ(const int ij){
   int i = ij%kpack, ki = i/2, spin_i = i%2;
   int j = ij/kpack, kj = j/2, spin_j = j%2;
   return (ki==kj)? 0.5 : 1.0;
}

// --- display ---
template <typename Tm>
void oper_display(oper_dict<Tm>& qops, const std::string sinfo, const int level=0){
   std::string oplist = "HCSABPQ";
   std::map<char,int> exist;
   std::string s(sinfo + ": ");
   for(const auto& key : oplist){
      if(qops.find(key) != qops.end()){ 
	 s += key;
	 s += ":"+std::to_string(qops[key].size())+ " ";
	 exist[key] = qops[key].size();
      }else{
	 exist[key] = 0;
      }
   }
   std::cout << s << std::endl;
   // print each operator
   if(level > 0){
      for(const auto& key : oplist){
         if(exist[key] > 0){
            std::cout << " " << key << ": ";
	    if(key == 'H' || key == 'C' || key == 'S'){
	       for(const auto& op : qops[key]){
	          std::cout << op.first << " ";
	       }
	    }else{
	       for(const auto& op : qops[key]){
	          auto pq = oper_unpack(op.first);
	          std::cout << "(" << pq.first << "," << pq.second << ") ";
	       }
	    }
	    std::cout << std::endl;
         }
      }
   } // level
}

} // ctns

#endif
