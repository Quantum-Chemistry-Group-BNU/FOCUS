#include <iostream>
#include <algorithm>
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

// copy constructor
onstate::onstate(const onstate& state){
   cout << "copy-c" << endl;	
   _size = state._size;
   _len  = state._len; 
   _repr = new long[state._len];
   copy(state._repr, state._repr+state._len, _repr);
}

// copy assignement constructor
onstate& onstate::operator =(const onstate& state){
   cout << "copy-=" << endl;	
   if(this != &state){
      _size = state._size;
      _len  = state._len;
      delete[] _repr;
      _repr = new long[_len];
      copy(state._repr, state._repr+_len, _repr);
   }
   return *this;
}

// move constructor
onstate::onstate(onstate&& state){
   cout << "move-c" << endl;	
   _size = state._size;
   _len  = state._len;
   _repr = state._repr;
   state._size = 0;
   state._len  = 0;
   state._repr = nullptr; // to avoid delete the pointed memory
}

// move assignement constructor
onstate& onstate::operator =(onstate&& state){
   cout << "move-=" << endl;	
   if(this != &state){
      _size == state._size;
      _len  == state._len;
      delete[] _repr;
      _repr = state._repr;
      state._size = 0;
      state._len  = 0;
      state._repr = nullptr; // to avoid delete the pointed memory
   }
   return *this;
}

// print (must declare fock::operator)
ostream& fock::operator <<(ostream& os, const onstate& state){
   os << state.to_string();
   return os;
}

// print 01 string
string onstate::to_string() const{
   string s; // string counts from left to right
   for(int i = _size-1; i >= 0; i--){
      s += (*this)[i] ? '1' : '0'; 
   }
   return s;
}

// print 2ab0 string
string onstate::to_string2() const{
   assert(_size%2 == 0); // only for even no. of basis function 
   string s; 
   int nodd,neven; 
   for(int i = _size/2-1; i >= 0; i--){
      nodd  = (*this)[2*i+1];
      neven = (*this)[2*i];
      if(nodd == 1 and neven == 1)
	 s += '2';
      else if(nodd == 0 and neven == 1)
	 s += 'a';
      else if(nodd == 1 and neven == 0)
	 s += 'b';
      else if(nodd == 0 and neven == 0)
	 s += '0';
   }
   return s;
}
