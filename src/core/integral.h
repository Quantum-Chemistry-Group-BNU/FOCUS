#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <unordered_map>
#include <cassert>
#include "../settings/global.h"
#include "tools.h"

namespace integral{

struct one_body{
   public:
      // constructor
      one_body(){};
      // copy/move
      one_body(const one_body& int1e){ // copy
         sorb = int1e.sorb;
         data = int1e.data;
      }
      one_body& operator =(const one_body& int1e){ // copy
         sorb = int1e.sorb;
         data = int1e.data;
         return *this;
      }
      one_body(one_body&& int1e){ // move
         sorb = int1e.sorb;
         data = move(int1e.data);   
      }
      one_body& operator =(one_body&& int1e){ // move
         sorb = int1e.sorb;
         data = move(int1e.data);
         return *this;
      }
      // core functions
      void print();
      // clear
      void clear(){ data.clear(); }
      // memsize
      double get_mem_space(){
         return global::mem_size(data.size());
      }
      // generic case: h1e[i,j]=[i|h|j]=[j|h|i]^*
      double get(const size_t i, const size_t j) const{
	 size_t key = tools::canonical_pair(i,j);
	 auto search = data.find(key);
	 if(search == data.end()){
	    return 0.0;
	 }else{
            if(i >= j){
	       return data.at(key); // at work with const
            }else{
	       return data.at(key); // complex conjugate in future
            }
	 }
      }
      void set(const size_t i, const size_t j, 
	       const double val){
	 size_t key = tools::canonical_pair(i,j);
         if(i >= j){
	    data[key] = val;
	 }else{
	    data[key] = val; // complex conjugate in future
	 }
      }
      // functionalities
      one_body get_AA() const; // [A|A]
      one_body get_BB() const; // [B|B]
      one_body get_BA() const; // [B|A],[A|B] (lower-triangular)
      // operations
      friend one_body operator +(const one_body& int1eA,
		      	         const one_body& int1eB);
   public:
      int sorb;
      std::unordered_map<size_t,double> data; // sparse representation
};

// redundant indexing function for 
// [ij|kl] = [ji|lk]^* = [kl|ij] = [lk|ji]^*
inline size_t canonical_quad(const size_t i, const size_t j,
			     const size_t k, const size_t l){
   size_t i1 = i+1;
   size_t i2 = i1*i1;
   return i*i*i2/4 + j*i2 + k*i1 + l;
}

inline std::tuple<size_t,size_t,size_t,size_t> inverse_quad(const size_t addr){
   size_t i = floor((sqrt(sqrt(addr*1.0)*8+1)-1)/2);
   size_t i1 = i+1;
   size_t i2 = i1*i1;
   size_t r1 = addr-i*i*i2/4;
   size_t j = floor(r1/i2);
   size_t r2 = r1-j*i2;
   size_t k = floor(r2/i1);
   size_t l = r2-k*i1;
   assert(canonical_quad(i,j,k,l)==addr);
   return make_tuple(i,j,k,l);
}

// return key,conj
inline std::pair<size_t,bool> packed_quad(const size_t i, 
					  const size_t j, 
	 				  const size_t k, 
					  const size_t l){
   auto ti = std::make_tuple(i,j,k,l);
   auto tk = std::make_tuple(k,l,i,j);
   auto tj = std::make_tuple(j,i,l,k);
   auto tl = std::make_tuple(l,k,j,i);
   if(ti>=tk && ti>=tj && ti>=tl) return make_pair(canonical_quad(i,j,k,l),false);
   if(tk>=ti && tk>=tj && tk>=tl) return make_pair(canonical_quad(k,l,i,j),false);
   if(tj>=ti && tj>=tk && tj>=tl) return make_pair(canonical_quad(j,i,l,k),true);
   if(tl>=ti && tl>=tk && tl>=tj) return make_pair(canonical_quad(l,k,j,i),true);
}

class two_body{
   public:
      // constructor
      two_body(){};
      // copy/move
      two_body(const two_body& int2e){ // copy
         sorb = int2e.sorb;
         data = int2e.data;
      }
      two_body& operator =(const two_body& int2e){ // copy
         sorb = int2e.sorb;
         data = move(int2e.data);
         return *this;
      }
      two_body(two_body&& int2e){ // move
         sorb = int2e.sorb;
         data = move(int2e.data);
      }
      two_body& operator =(two_body&& int2e){ // move
         sorb = int2e.sorb;
         data = move(int2e.data);
         return *this;
      }
      // core functions
      void print();
      // clear
      void clear(){ data.clear(); }
      // memsize
      double get_mem_space(){
         return global::mem_size(data.size());
      }
      // [ij|kl] = [ji|lk]^* = [kl|ij] = [lk|ji]^*
      double get(const size_t i, const size_t j, 
		 const size_t k, const size_t l) const{
         auto p = packed_quad(i,j,k,l);
         size_t key = p.first;
	 bool ifconj = p.second;
	 auto search = data.find(key);
	 if(search == data.end()){
	    return 0.0;
	 }else{
            if(!ifconj){
	       return data.at(key);
            }else{
	       return data.at(key); // complex conjugate in future
            }
	 }
      }
      void set(const size_t i, const size_t j, 
	       const size_t k, const size_t l, 
	       const double val){
         auto p = packed_quad(i,j,k,l);
         size_t key = p.first;
	 bool ifconj = p.second;
	 if(!ifconj){
	    data[key] = val;
	 }else{
	    data[key] = val; // complex conjugate in future
	 }
      }
      // functionalities 
      two_body get_AAAA() const; // [AA|AA]
      two_body get_BBBB() const; // [BB|BB]
      two_body get_BBAA() const; // [BB|AA],[AA|BB]
      two_body get_BAAA() const; // [BA|AA],[AB|AA],[AA|BA],[AA|AB]
      two_body get_BABA() const; // [BA|BA],[BA|AB],[AB|BA],[AB|AB]
      two_body get_BBBA() const; // [BB|BA],[BB|AB],[BA|BB],[AB|BB]
      // operations
      friend two_body operator +(const two_body& int2eA,
		      	         const two_body& int2eB);
   public:
      int sorb;
      std::unordered_map<size_t,double> data; // sparse representation
      vector<double> J, K, Q; // [ii|jj],[ij|ji],<ij||ij>
};

void read_fcidump(two_body& int2e, one_body& int1e, double& ecore,
		  std::string fcidump="FCIDUMP");

} // integral

#endif
