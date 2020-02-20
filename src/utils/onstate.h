#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <string>
#include <cassert>

using namespace std;

namespace fock{

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
      onstate(const int n): _size(n){
         assert(n>0);
         _len = (n-1)/64+1;
	  _repr = new long[_len];
         for(int i=0; i<_len; i++)
	    _repr[i] = 0;
      }
      onstate(const string& on);
      // destructors
      ~onstate() { delete[] _repr; }
      // assignment constructor
      onstate& operator =(const onstate& state);
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
      // count the number of 1 in ontate
      int Ne() const;
      /*
      int Na();
      int Nb();
      int Nd();
      */
      // print
      friend ostream& operator <<(ostream& os, const onstate& state);
   public:
      int _size;
      int _len;
      long* _repr;
};

}

#endif
