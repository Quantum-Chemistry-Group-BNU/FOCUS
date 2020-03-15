#ifndef ONSTATE_H
#define ONSTATE_H

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <tuple>

namespace fock{

// assuming i < 64, return 0[i=0],1[i=1],11[i=2],...
// must be inlined in header
inline unsigned long get_ones(const int& n){
   return (1ULL<<n) - 1ULL; // parenthesis must be added due to priority
}

// count the number of nonzero bits
inline int popcnt(unsigned long x)
{
#ifdef GNU
   return __builtin_popcount(x);
#else
   // https://en.wikipedia.org/wiki/Hamming_weight
   const uint64_t m1  = 0x5555555555555555; //binary: 0101...
   const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
   const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
   const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
   x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
   x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
   x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
   return (x * h01) >> 56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
#endif
}

class bit_proxy{
   public:
      // constructor
      bit_proxy(unsigned long& dat, unsigned long mask): _dat(dat), _mask(mask) {}
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
      unsigned long& _dat;
      unsigned long _mask;
};

class onstate{
   public:
      
      // constructors
      onstate(): _size(0), _len(0), _repr(nullptr) {};
      onstate(const int n){
         assert(n>0);
	 _size = n;
         _len = (n-1)/64+1;
	 _repr = new unsigned long[_len];
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
      // from "01011"
      onstate(const std::string& on); 
      // merge two states with different spins - neglect phases
      onstate(const onstate& state_a, const onstate& state_b);

      // core functions
      int len() const{ return _len; }
      int size() const{ return _size; }
      unsigned long repr(const int i) const{ return _repr[i]; }
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
	    if(_repr[i] < state._repr[i]){ 
	       return true;
	    }else if(_repr[i] > state._repr[i]){
	       return false;
	    }else if(_repr[i] == state._repr[i]){
	       continue;
	    }
	 }
	 return false; // occurs when *this == state
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
	 unsigned long mask = 0x5555555555555555;
         for(int i=0; i<_len; i++){
 	    unsigned long even = _repr[i] & mask;
            ne += popcnt(even);
         }
         return ne;
      }
      int nelec_b() const{
         int ne = 0; // 0xA = 1010 [count from left]
	 unsigned long mask = 0xAAAAAAAAAAAAAAAA;
         for(int i=0; i<_len; i++){
 	    unsigned long odd = _repr[i] & mask;
            ne += popcnt(odd);
         }
         return ne;
      }
      
      // creation/annihilation operators related subroutines
      //
      // phase to convert |ON>=f[odd,even]|odd>|even>
      // f[odd,even] = sum_{i=0}^{n-1} f[2i+1] (sum_{j=0}^{i} f[2j])
      // n=3: f1*f0
      //      f3*(f2+f0)
      //      f5*(f4+f2+f0) 
      int parity_odd_even() const{
         int ff = 0;
	 for(int i=0; i<_size/2; i++){
	    if(!(*this)[2*i+1]) continue;
	    for(int j=0; j<=i; j++){
	       if((*this)[2*j]) ff += 1;
	    }
	 }
	 return -2*(ff%2)+1;
      }
      // parity: = sum_k(-1)^fk for k in [0,n) 
      int parity(const int& n) const{
	 assert(n>=0 && n<_size);     
         int nonzero = 0;
	 for(int i=0; i<n/64; i++){
            nonzero += popcnt(_repr[i]);
	 }
	 nonzero += popcnt((_repr[n/64] & get_ones(n%64)));
	 return -2*(nonzero%2)+1;
      }
      // parity: = sum_k(-1)^fk for k in [start,end) 
      int parity(const int& start, const int& end) const{
	 assert(start>=0 && start<_size);
         assert(end>=0 && start<_size);
         assert(start<end);
         unsigned long mask = get_ones(start%64);
         unsigned long res = _repr[start/64] & mask;
         int nonzero = -popcnt(res);
         for(int i=start/64; i<end/64; i++)
	    nonzero += popcnt(_repr[i]);
         mask = get_ones(end%64);
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
      
      // connection for hamiltonian
      int diff_num(const onstate& state) const{
         int ndiff = 0;
	 for(int i=0; i<_len; i++){
            ndiff += popcnt(_repr[i]^state._repr[i]);
	 }
         return ndiff;
      }
      // connection type: (2,0) for <bra|c1^+c2^+|ket>
      // 	       also implies <bra|c1^+c2^+k^+k|ket> 
      std::pair<int,int> diff_type(const onstate& ket) const{
         unsigned long idiff,icre,iann;
	 std::pair<int,int> p(0,0);
         for(int i=_len-1; i>=0; i--){
            idiff = _repr[i] ^ ket._repr[i];
            icre = idiff & _repr[i];
            iann = idiff & ket._repr[i];
            p.first  += popcnt(icre);
            p.second += popcnt(iann);
         }
         return p; // ndiff = p.first + p.second
      }
      // orbital difference
      void diff_orb(const onstate& ket,
		    std::vector<int>& cre,
		    std::vector<int>& ann) const;
      void diff_orb(const onstate& ket,
      		    int* cre, int* ann) const;
      // even strings 
      onstate get_even() const{
         assert(_size%2 == 0);
	 onstate state(_size/2);
	 for(int i=0; i<_size/2; i++){
	    if((*this)[2*i]) state[i] = 1; 
	 }
         return state;
      }
      // odd strings 
      onstate get_odd() const{
         assert(_size%2 == 0);
	 onstate state(_size/2);
	 for(int i=0; i<_size/2; i++){
	    if((*this)[2*i+1]) state[i] = 1; 
	 }
         return state;
      }

      // occupied, virtual orbital lists 
      void get_olst(std::vector<int>& olst) const;
      void get_olst(int* olst) const;
      void get_vlst(std::vector<int>& vlst) const;
      void get_vlst(int* vlst) const;
      
      // number of spatial orbitals
      int norb() const{ return _size/2; }
      // number of singly occupied (seniority number)
      int norb_single() const{
         unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     unsigned long tmp = ((_repr[i]&even)<<1) ^ (_repr[i]&odd); // xor
	     num += popcnt(tmp);
	 }
	 return num;
      }
      // number of doubly occupied
      int norb_double() const{
         unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     unsigned long tmp = ((_repr[i]&even)<<1) & (_repr[i]&odd); // and
	     num += popcnt(tmp);
	 }
	 return num;
      }
      // number of vacant
      int norb_vacant() const{
         unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int num = 0;
	 for(int i=_len-1; i>=0; i--){
	     unsigned long tmp = ((_repr[i]&even)<<1) | (_repr[i]&odd); // or
	     num -= popcnt(tmp);
	 }
	 num += norb();
	 return num;
      }
      
      // kramers symmetry related functions
      bool has_single() const{
         unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 for(int i=_len-1; i>=0; i--){
	    if( ((_repr[i]&even)<<1) != (_repr[i]&odd) ) return true;
	 }
	 return false;
      }
      // standard representative for {|state>,K|state>}
      bool is_standard() const{
         if(!has_single()) return true;
	 unsigned long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 for(int i=_len-1; i>=0; i--){
	    unsigned long flipped = ((_repr[i]&even)<<1) + ((_repr[i]&odd)>>1);
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
      unsigned long* _repr;
};

} // fock

#endif
