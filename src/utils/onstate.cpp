#include <iostream>
#include "onstate.h"

using namespace fock;

// constructor from '010011' strings
onstate::onstate(const string& s){
   _size = s.size();
   _len = (_size-1)/64+1;
   _repr = new long[_len];
   for(int i=0; i<_len; i++)
      _repr[i] = 0;
   for(int i=0; i<_size; i++){
      (*this)[i] = s[_size-1-i] == '1' ? 1 : 0;
   }
}

// print (must declare fock::operator)
ostream& fock::operator <<(ostream& os, const onstate& state){
   string s; // string counts from left to right
   for(int i = state.size()-1; i >= 0; i--){
      s += state[i] ? '1' : '0'; 
   }
   os << s;
   return os;
}

// assignement constructor
onstate& onstate::operator =(const onstate& state){
   if(this != &state){
      assert(_size == state.size());
      for(int i=0; i<_len; i++){
	 _repr[i] = state._repr[i];
      }
   }
   return *this;
}

int onstate::Ne() const{
   int ne = 0;
   for(int i=0; i<_len; i++){
      ne += popcnt(_repr[i]);
   }
   return ne;
}
