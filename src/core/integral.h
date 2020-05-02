#ifndef INTEGRAL_H
#define INTEGRAL_H

#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <cassert>
#include "../settings/global.h" // mem_size

namespace integral{

void save_text_sym1e(const std::vector<double>& data,
		     const std::string& fname, 
		     const int prec=12);

struct one_body{
   public:
      // constructor
      one_body(){};
      one_body(const int n) : sorb(n){
         assert(n > 0);
         data.resize(n*(n+1)/2,0.0); 
      };
      // init_memory
      void init_mem(){
	 assert(sorb > 0);
	 data.resize(sorb*(sorb+1)/2,0.0);
      }
      // memsize
      double get_mem(){
         return global::mem_size(data.size());
      }
      // core functions
      void print();
      // generic case: h1e[i,j]=[i|h|j]=[j|h|i]^*
      double get(const size_t i, const size_t j) const{
	 size_t key = std::max(i,j)*(std::max(i,j)+1)/2 + std::min(i,j); 
         if(i >= j){
	    return data[key]; 
         }else{
	    return data[key]; // complex conjugate in future
         }
      }
      void set(const size_t i, const size_t j, 
	       const double val){
	 size_t key = std::max(i,j)*(std::max(i,j)+1)/2 + std::min(i,j); 
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
      std::vector<double> data;
};

// redundant indexing function for 
// [ij|kl] = [ji|lk]^* = [kl|ij] = [lk|ji]^*
// scheme: for canonical index [ij|kl] store [ij|**] 
// to simplify indexing, asymptotically O(n4/4) 
inline size_t canonical_quad(const size_t i, const size_t j,
			     const size_t k, const size_t l){
   size_t i1 = i+1;
   size_t i2 = i1*i1;
   return i*i*i2/4 + j*i2 + k*i1 + l;
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
   if(ti>=tk && ti>=tj && ti>=tl) return std::make_pair(canonical_quad(i,j,k,l),false);
   if(tk>=ti && tk>=tj && tk>=tl) return std::make_pair(canonical_quad(k,l,i,j),false);
   if(tj>=ti && tj>=tk && tj>=tl) return std::make_pair(canonical_quad(j,i,l,k),true);
   if(tl>=ti && tl>=tk && tl>=tj) return std::make_pair(canonical_quad(l,k,j,i),true);
   exit(1);
}

// not used here
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
   return std::make_tuple(i,j,k,l);
}

struct two_body{
   public:
      // constructor
      two_body(){};
      two_body(const int n) : sorb(n){
         assert(n > 0);
	 size_t p = n*(n+1)/2; 
         data.resize(p*p,0.0);
      };
      // init_memory
      void init_mem(){
	 assert(sorb > 0);
	 size_t p = sorb*(sorb+1)/2;
         data.resize(p*p,0.0);
      }
      // memsize
      double get_mem(){
         return global::mem_size(data.size());
      }
      // core functions
      void print();
      // [ij|kl] = [ji|lk]^* = [kl|ij] = [lk|ji]^*
      double get(const size_t i, const size_t j, 
		 const size_t k, const size_t l) const{
         auto p = packed_quad(i,j,k,l);
         //size_t key = p.first;
	 //bool ifconj = p.second;
         if(!p.second){
	    return data[p.first];
         }else{
	    return data[p.first]; // complex conjugate in future
         }
      }
      // <ij||kl> = [ik|jl]-[il|jk]
      double getAnti(const size_t i, const size_t j,
		     const size_t k, const size_t l) const{
	 return this->get(i,k,j,l) - this->get(i,l,j,k);
      }
      void set(const size_t i, const size_t j, 
	       const size_t k, const size_t l, 
	       const double val){
         auto p = packed_quad(i,j,k,l);
         //size_t key = p.first;
	 //bool ifconj = p.second;
	 if(!p.second){
	    data[p.first] = val;
	 }else{
	    data[p.first] = val; // complex conjugate in future
	 }
      }
      // functionalities 
      two_body get_AAAA() const; // [AA|AA]
      two_body get_BBBB() const; // [BB|BB]
      two_body get_BBAA() const; // [BB|AA],[AA|BB]
      two_body get_BAAA() const; // [BA|AA],[AB|AA],[AA|BA],[AA|AB]
      two_body get_BABA() const; // [BA|BA],[BA|AB],[AB|BA],[AB|AB]
      two_body get_BBBA() const; // [BB|BA],[BB|AB],[BA|BB],[AB|BB]
      // J=[ii|jj], K=[ij|ji], Q=<ij||ij>
      void set_JKQ(){
	 J.resize(sorb*(sorb+1)/2,0.0);
	 for(int i=0; i<sorb; i++){
	    for(int j=0; j<=i; j++){
	       int ij = i*(i+1)/2+j;
	       J[ij] = this->get(i,i,j,j);
	    }
	 }
	 K.resize(sorb*(sorb+1)/2,0.0);
	 for(int i=0; i<sorb; i++){
	    for(int j=0; j<=i; j++){
	       int ij = i*(i+1)/2+j;
	       K[ij] = this->get(i,j,j,i);
	    }
	 }
	 Q.resize(sorb*(sorb-1)/2,0.0);
	 for(int i=0; i<sorb; i++){
	    for(int j=0; j<i; j++){
	       int ij1 = i*(i-1)/2+j;
	       int ij2 = i*(i+1)/2+j;
	       Q[ij1] = J[ij2] - K[ij2];
	    }
	 }
      }
      // operations
      friend two_body operator +(const two_body& int2eA,
		      	         const two_body& int2eB);
   public:
      int sorb;
      std::vector<double> data; // [ij|kl] 
      std::vector<double> J, K, Q; // [ii|jj],[ij|ji],<ij||ij>
};

void read_fcidump(two_body& int2e, one_body& int1e, double& ecore,
		  std::string fcidump, const int type=0);

} // integral

#endif
