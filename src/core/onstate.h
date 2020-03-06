#ifndef ONSTATE_H
#define ONSTATE_H

#include <iostream>
#include <cassert>
#include <string>
#include <vector>

namespace fock{

// assuming i < 64, return 0[i=0],1[i=1],11[i=2],...
// must be inlined in header
inline long allones(const int& n){
   return (1ULL<<n) - 1ULL; // parenthesis must be added due to priority
}

inline int popcnt(long x)
{
   x = (x & 0x5555555555555555ULL) + ((x >> 1) & 0x5555555555555555ULL);
   x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
   x = (x & 0x0F0F0F0F0F0F0F0FULL) + ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
   return (x * 0x0101010101010101ULL) >> 56;
}

class bit_proxy{
   public:
      // constructor
      bit_proxy(long& dat, long mask): _dat(dat), _mask(mask) {}
      // reading access is realized with a conversion to bool where
      operator bool() const { return _dat & _mask; }
      // enable assignement using side effect of this function
      bit_proxy& operator =(bool b){
         if(b){
   	    _dat |= _mask;
	 }else{ 
 	    _dat &= ~_mask;	
	 }	 
	 return *this; 
      }
   private:
      long& _dat;
      long _mask;
};

class onstate{
   public:
      // constructors
      onstate(): _size(0), _len(0), _repr(nullptr) {};
      onstate(const int n){
         assert(n>0);
	 _size = n;
         _len = (n-1)/64+1;
	 _repr = new long[_len];
         for(int i=0; i<_len; i++)
	    _repr[i] = 0;
      }
      // destructors
      ~onstate(){ delete[] _repr; }
      // copy constructor
      onstate(const onstate& state);
      // copy assignment 
      onstate& operator =(const onstate& state);
      // move constructor - move resources to this
      onstate(onstate&& state);
      // move assignement [no const, since it will be modified] 
      onstate& operator =(onstate&& state);
      // special constructors
      onstate(const std::string& on); // from "01011"
      onstate(const onstate& state_a, const onstate& state_b); // merge
      // core functions
      int len() const{ return _len; }
      int size() const{ return _size; }
      // getocc: only for the case where onstate is read-only (const) 
      bool operator [](const int i) const{ 
	  assert(i < _size);
	  // default 1<<i%64 will incur error, since 1 is int - 32bit
	  return _repr[i/64] & (1ULL << i%64);
      }
      // setocc
      bit_proxy operator [](const int i){
	  assert(i < _size);
	  return {_repr[i/64] , 1ULL << i%64}; 
      }
      // print
      std::string to_string() const;
      std::string to_string2() const;
      friend std::ostream& operator <<(std::ostream& os, const onstate& state);
      // comparison [from high position] 
      bool operator <(const onstate& state) const{
         for(int i=_len-1; i>=0; i--){
	    if(_repr[i] < state._repr[i]) 
	       return true;
	    else
	       break;
	 }
	 return false; 
      }
      bool operator ==(const onstate& state) const{
         for(int i=_len-1; i>=0; i--){
	    if(_repr[i] != state._repr[i]) return false;
	 } 
	 return true;
      }
      // count the number of 1 in ontate
      int nelec() const{
         int ne = 0;
         for(int i=0; i<_len; i++){
            ne += popcnt(_repr[i]);
         }
         return ne;
      }
      int nelec_a() const{
         int ne = 0; // 0x5 = 0101 [count from left]
	 long mask = 0x5555555555555555;
         for(int i=0; i<_len; i++){
 	    long even = _repr[i] & mask;
            ne += popcnt(even);
         }
         return ne;
      }
      int nelec_b() const{
         int ne = 0; // 0xA = 1010 [count from left]
	 long mask = 0xAAAAAAAAAAAAAAAA;
         for(int i=0; i<_len; i++){
 	    long odd = _repr[i] & mask;
            ne += popcnt(odd);
         }
         return ne;
      }
      // creation/annihilation operators subroutines
      // parity: = sum_k(-1)^fk for k in [0,n) 
      int parity(const int& n) const{
         int nonzero = 0;
	 for(int i=0; i<n/64; i++){
            nonzero += popcnt(_repr[i]);
	 }
	 nonzero += popcnt((_repr[n/64] & allones(n%64)));
	 return -2*(nonzero%2)+1;
      }
      // parity: = sum_k(-1)^fk for k in [start,end) 
      int parity(const int& start, const int& end) const{
         long mask = allones(start%64);
         long res = _repr[start/64] & mask;
         int nonzero = -popcnt(res);
         for(int i=start/64; i<end/64; i++)
	    nonzero += popcnt(_repr[i]);
         mask = allones(end%64);
         res = _repr[end/64] & mask;	 
	 nonzero += popcnt(res);
	 return -2*(nonzero%2)+1;
      }
      // i^+|state>
      std::pair<int,onstate> cre(const int& i) const{
         int fac;
	 onstate res;
	 if((*this)[i]){
	    fac = 0; // vanishing result
         }else{
	    res = *this;
	    res[i] = 1;
	    fac = res.parity(i);
	 }
	 return std::make_pair(fac,std::move(res));
      }
      // i|state>
      std::pair<int,onstate> ann(const int& i) const{
         int fac;
	 onstate res;
	 if((*this)[i]){
	    res = *this;
	    res[i] = 0;
	    fac = res.parity(i);
	 }else{
	    fac = 0;
	 }
	 return std::make_pair(fac,std::move(res));
      }
      // hamiltonian
      int num_diff(const onstate& state) const{
         int ndiff = 0;
	 for(int i=0; i<_len; i++){
            ndiff += popcnt(_repr[i]^state._repr[i]);
	 }
         return ndiff/2;
      }
      // orbital difference
      friend void orb_diff(const onstate& bra,
		      	   const onstate& ket,
			   std::vector<int>& cre,
			   std::vector<int>& ann);
      // we assume the no. of electrons are the same,
      // to avoid checking this condition. 
      bool if_Hconnected(const onstate& state) const{
         return num_diff(state) <= 2;
      }
      // occupied-virtual lists 
      void get_occ(std::vector<int>& olst) const{
         for(int i=0; i<_size; i++){
	    if((*this)[i])
	       olst.push_back(i);
	 }
      }
      void get_vir(std::vector<int>& vlst) const{
         for(int i=0; i<_size; i++){
	    if(!(*this)[i])
	       vlst.push_back(i);
	 }
      }
      void get_occvir(std::vector<int>& olst, 
		      std::vector<int>& vlst) const{
         for(int i=0; i<_size; i++){
	    if((*this)[i])
	       olst.push_back(i);
	    else
	       vlst.push_back(i);
	 }
      }
      // number of spatial orbitals
      int norb() const{ return _size/2; }
      // number of singly occupied (seniority number)
      int norb_single() const{
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     long tmp = ((_repr[i]&even)<<1) ^ (_repr[i]&odd); // xor
	     num += popcnt(tmp);
	 }
	 return num;
      }
      // number of doubly occupied
      int norb_double() const{
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     long tmp = ((_repr[i]&even)<<1) & (_repr[i]&odd); // and
	     num += popcnt(tmp);
	 }
	 return num;
      }
      // number of vacant
      int norb_vacant() const{
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     long tmp = ((_repr[i]&even)<<1) | (_repr[i]&odd); // or
	     num -= popcnt(tmp);
	 }
	 num += norb();
	 return num;
      }
      // kramers symmetry related functions
      bool has_single() const{
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 for(int i=_len-1; i>=0; i--){
	    if( ((_repr[i]&even)<<1) != (_repr[i]&odd) ) return true;
	 }
	 return false;
      }
      // standard representative for {|state>,K|state>}
      bool is_standard() const{
         if(!has_single()) return true;
	 long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 for(int i=_len-1; i>=0; i--){
	    long flipped = ((_repr[i]&even)<<1) + ((_repr[i]&odd)>>1);
	    if(_repr[i] < flipped) 
               return false;
	    // define the det with max integer rep as standard
            else if(_repr[i] > flipped) 
 	       return true;
	 }
	 std::cout << "error in onstate.is_standard" << std::endl;
	 std::cout << *this << std::endl;
 	 exit(1);
      }
      // flip K{|a>,|b>}={|b>,-|a>}
      onstate flip() const{
         unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 onstate state(_size); 
	 for(int i=0; i<_len; i++){
	    state._repr[i] = ((_repr[i]&even)<<1) + ((_repr[i]&odd)>>1);
	 }
	 return state;
      }
      // return standard representative for {|state>,K|state>}
      onstate make_standard() const{
	 if(is_standard())
	    return *this;
	 else
	    return flip();
      }
      // K|state>=|state'>(-1)^{sum[na*nb+nb]} (na*nb=nd)
      int parity_flip() const{
	 return -2*((norb_double()+nelec_b())%2)+1;
      }
   private:
      int _size;
      int _len;
      long* _repr;
};

// compare two states
void orb_diff(const onstate& bra, const onstate& ket,
	      std::vector<int>& cre, std::vector<int>& ann);

}

#endif
