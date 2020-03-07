#include <iostream>
#include <algorithm>
#include "onstate.h"
#include "../settings/global.h"

using namespace std;
using namespace fock;

// constructor from '010011' strings
onstate::onstate(const string& s){
   _size = s.size();
   _len = (_size-1)/64+1;
   _repr = new unsigned long[_len];
   fill_n(_repr, _len, 0); // initialize, otherwise some unused bits will not be 0.
   for(int i=0; i<_size; i++){
      (*this)[i] = s[_size-1-i] == '1' ? 1 : 0;
   }
}

// merge two states with different spins - neglect phases
onstate::onstate(const onstate& state_a, const onstate& state_b){
   assert(state_a._size == state_b._size);
   _size = 2*state_a._size;
   _len = (_size-1)/64+1;
   _repr = new unsigned long[_len];
   fill_n(_repr, _len, 0); // initialize, otherwise some unused bits will not be 0.
   for(int ia=0; ia<state_a._size; ia++){
      (*this)[2*ia] = state_a[ia];
      (*this)[2*ia+1] = state_b[ia]; 
   }
}

// copy constructor
onstate::onstate(const onstate& state){
   if(global::print_level>3) cout << "copy c" << endl;	
   _size = state._size;
   _len  = state._len;
   _repr = new unsigned long[_len];
   copy_n(state._repr, _len, _repr);
}

// copy assignement 
onstate& onstate::operator =(const onstate& state){
   if(global::print_level>3) cout << "copy =" << endl;	
   if(this != &state){
      if(global::print_level>3) cout << "copyd=" << endl;	   
      _size = state._size;
      _len  = state._len;
      delete[] _repr;
      _repr = new unsigned long[_len];
      copy_n(state._repr, _len, _repr);
   }
   return *this;
}

// move constructor
onstate::onstate(onstate&& state){
   if(global::print_level>3) cout << "move c" << endl;	
   _size = state._size;
   _len  = state._len;
   _repr = state._repr;
   state._size = 0;
   state._len  = 0;
   state._repr = nullptr; // to avoid delete the pointed memory
}

// move assignement constructor
onstate& onstate::operator =(onstate&& state){
   if(global::print_level>3) cout << "move =" << endl;	
   if(this != &state){
      if(global::print_level>3) cout << "moved=" << endl;
      _size = state._size;
      _len  = state._len;
      delete[] _repr; // release memory that _repr hold
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
      if(nodd == 1 && neven == 1)
	 s += '2';
      else if(nodd == 0 && neven == 1)
	 s += 'a';
      else if(nodd == 1 && neven == 0)
	 s += 'b';
      else if(nodd == 0 && neven == 0)
	 s += '0';
   }
   return s;
}

// compute orbital difference: 
//
// the convention is p1>p2>...>pm, q1>q2>...>qn
// such that <bra|p1^+...pm^+ qn...q1|ket>=sgn(bra,p1)...sgn(bra,pm)
// 					  *sgn(ket,q1)...sgn(ket,qn)		
//
// derivation: |Phi_common> (onstate) = pm*...*p1|bra>*sgn(bra)
// 			 	      = qn*...*q1|ket>*sgn(ket)
// then <bra|p1^+...pm^+*pn*...*p1|bra> = <Phi|Phi> = 1
//      <bra|p1^+...pm^+*qn*...*q1|ket>*sgn(bra)*sgn(ket) = 1
// which leads to the above result.
void fock::diff_orb(const onstate& bra, const onstate& ket,
		    vector<int>& cre, vector<int>& ann){
   unsigned long idiff,icre,iann;
   // from higher position
   for(int i=bra._len-1; i>=0; i--){
      idiff = bra._repr[i] ^ ket._repr[i];
      icre = idiff & bra._repr[i];
      iann = idiff & ket._repr[i];
      for(int j=63; j>=0; j--)
         if(icre & 1ULL<<j) cre.push_back(i*64+j);
      for(int j=63; j>=0; j--)
	 if(iann & 1ULL<<j) ann.push_back(i*64+j);
   }
}

// connection type
pair<int,int> fock::diff_type(const onstate& bra, 
		      	      const onstate& ket){
   unsigned long idiff,icre,iann;
   pair<int,int> p(0,0);
   // from higher position
   for(int i=bra._len-1; i>=0; i--){
      idiff = bra._repr[i] ^ ket._repr[i];
      icre = idiff & bra._repr[i];
      iann = idiff & ket._repr[i];
      p.first  += fock::popcnt(icre);
      p.second += fock::popcnt(iann);
   }
   return p;
}
