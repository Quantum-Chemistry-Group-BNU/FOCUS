#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <string>
#include <cassert>

using namespace std;

namespace fock{

// assuming i < 64, return 0[i=0],1[i=1],11[i=2],...
// must be inlined in header
inline long allones(const int& n){
   long one = 1;
   return (one<<n) - one; // parenthesis must be added due to priority
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
      onstate(const string& on);
      // destructors
      ~onstate(){ delete[] _repr; }
      // copy constructor
      onstate(const onstate& state);
      // copy assignment constructor
      onstate& operator =(const onstate& state);
      // move constructor - move resources to this
      onstate(onstate&& state);
      // move assignement [no const, since it will be modified] 
      onstate& operator =(onstate&& state);
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
	  return {_repr[i/64],1ULL << i%64}; 
      }
      // print
      string to_string() const;
      string to_string2() const;
      friend ostream& operator <<(ostream& os, const onstate& state);
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
      bool operator =(const onstate& state) const{
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
      // hamiltonian
      int num_diff(const onstate& state) const{
         int ndiff = 0;
	 for(int i=0; i<_len; i++)
            ndiff += popcnt(_repr[i]^state._repr[i]);
	 return ndiff;
      }
      // we assume the no. of electrons are the same,
      // to avoid checking this condition. 
      bool if_Hconnected(const onstate& state) const{
         return this->num_diff(state) <= 4;
      }
      // creation/annihilation operators subroutines
      // parity: =0, even; =1, odd. 
      int parity(const int& n){
         int nonzero = 0;
	 for(int i=0; i<n/64; i++){
            nonzero += popcnt(_repr[i]);
	 }
	 nonzero += popcnt((_repr[n/64] & allones(n%64)));
	 return -2*(nonzero%2)+1;
      }
      int parity(const int& start, const int& end){
	 int ista = start%64;
         long mask = allones(ista);
         long res = _repr[start/64] & mask;
         int nonzero = -popcnt(res)-(*this)[ista];
         for(int i=start/64; i<end/64; i++)
	    nonzero += popcnt(_repr[i]);
         mask = allones(end%64);
         res = _repr[end/64] & mask;	 
	 nonzero += popcnt(res);
	 return -2*(nonzero%2)+1;
      }
      // i^+|state>
      pair<int,onstate> cre(const int& i) const{
         int fac;
	 onstate res;
	 if((*this)[i]){
	    fac = 0; // vanishing result
         }else{
	    res = *this;
	    res[i] = 1;
	    fac = res.parity(i);
	 }
	 return make_pair(fac,move(res));
      }
      // i|state>
      pair<int,onstate> ann(const int& i) const{
         int fac;
	 onstate res;
	 if((*this)[i]){
	    res = *this;
	    res[i] = 0;
	    fac = res.parity(i);
	 }else{
	    fac = 0;
	 }
	 return make_pair(fac,move(res));
      }



      // kramers symmetry related functions
      bool has_unpaired(){
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 for(int i=_len-1; i>=0; i--){
	    if( ((_repr[i]&even)<<1) != (_repr[i]&odd) ) return true;
	 }
	 return false;
      }
      int num_unpaired(){
         long even = 0x5555555555555555, odd = 0xAAAAAAAAAAAAAAAA;
	 int unpaired = 0;
	 for(int i=_len-1; i>=0; i--){
	     long single = ((_repr[i]&even)<<1) ^ (_repr[i]&odd);
	     unpaired += popcnt(single);
	 }
	 return unpaired;
      }
      // isStandard
      // flipAlphaBeta
      // parityOfFlipAlphaBeta
      // makeStandard
      //
      // double Energy(oneInt& I1, twoInt& I2, double& coreE);
      // CItype Hij(Determinant& bra, Determinant& ket, oneInt& I1, twoInt& I2, double& coreE, size_t& orbDiff);

   private:
      int _size;
      int _len;
      long* _repr;
};

}

#endif
