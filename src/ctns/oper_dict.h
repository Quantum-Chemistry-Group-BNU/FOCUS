#ifndef OPER_DICT_H
#define OPER_DICT_H

#include <map>
#include <tuple>
#include "qtensor/qtensor.h"
#include "../core/serialization.h"

namespace ctns{

// debug options
const bool debug_oper_dict = false;
extern const bool debug_oper_dict;

const bool debug_oper_io = false;
extern const bool debug_oper_io;

const bool debug_oper_para = false;
extern const bool debug_oper_para;

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

// --- distribution ---
inline int distribute2(const int index, const int size){
   auto pq = oper_unpack(index);
   int p = pq.first, q = pq.second;
   assert(p <= q);
   return (p == q)? p%size : (q*(q-1)/2+p)%size;
}

inline int oper_num_opA(const int cindex1_size, const bool& ifkr){
   int k = ifkr? cindex1_size : cindex1_size/2;
   int num = ifkr? k*k : k*(2*k-1);
   return num;
}

inline int oper_num_opB(const int cindex1_size, const bool& ifkr){
   int k = ifkr? cindex1_size : cindex1_size/2;
   int num = ifkr? k*(k+1) : k*(2*k+1);
   return num;
}

inline std::vector<int> oper_index_opA(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> aindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
	 if(p1 < q1){ 
            aindex.push_back( oper_pack(p1,q1) );
            if(ifkr) aindex.push_back( oper_pack(p1,q1+1) );
	 }else if(p1 == q1){
	    if(ifkr) aindex.push_back( oper_pack(p1,p1+1) );
	 }
      }
   }
   int kc = cindex1.size();
   assert(aindex.size() == oper_num_opA(kc,ifkr));
   return aindex;
}

inline std::vector<int> oper_index_opB(const std::vector<int>& cindex1, const bool& ifkr){
   std::vector<int> bindex;
   for(int p1 : cindex1){
      for(int q1 : cindex1){
         if(p1 <= q1){
            bindex.push_back( oper_pack(p1,q1) );
	    if(ifkr) bindex.push_back( oper_pack(p1,q1+1) );
	 }
      }
   }
   int kc = cindex1.size();
   assert(bindex.size() == oper_num_opB(kc,ifkr));
   return bindex;
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

// --- oper_dict: container for operators --- 
template <typename Tm>
using oper_map = std::map<int,qtensor2<Tm>>;

template <typename Tm>
class oper_dict{
private:
   // serialize
   friend class boost::serialization::access;
   template<class Archive>
   void serialize(Archive & ar, const unsigned int version){
      ar & cindex & ops;
   }
public:
   // constructor
   oper_dict(){
      oper_map<Tm> opm; // empty operator map 
      ops = {{'C',opm},{'A',opm},{'B',opm},
	     {'S',opm},{'P',opm},{'Q',opm},
	     {'H',opm}};
   }
   // access
   const oper_map<Tm>& operator()(const char key) const{
      return ops.at(key);
   }
   oper_map<Tm>& operator()(const char key){
      return ops[key];      
   }
   // print
   void print(const std::string name, const int level=0) const{
      std::cout << " " << name << " : size[cindex]=" << cindex.size(); 
      // count no. of operators in each class
      std::string oplist = "CABHSPQ";
      std::map<char,int> exist;
      std::string s = " nops=";
      for(const auto& key : oplist){
         if(ops.find(key) != ops.end()){ 
            s += key;
            s += ":"+std::to_string(ops.at(key).size())+ " ";
            exist[key] = ops.at(key).size();
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
                  for(const auto& op : ops.at(key)){
                     std::cout << op.first << " ";
                  }
               }else{
                  for(const auto& op : ops.at(key)){
                     auto pq = oper_unpack(op.first);
                     std::cout << "(" << pq.first << "," << pq.second << ") ";
                  }
               }
               std::cout << std::endl;
            }
         }
      } // level
   }
public:
   std::vector<int> cindex;
private:
   std::map<char,oper_map<Tm>> ops;
};

} // ctns

#endif
