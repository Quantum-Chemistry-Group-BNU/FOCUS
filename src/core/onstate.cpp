#include <iostream>
#include <algorithm>
#include "onstate.h"

using namespace std;
using namespace fock;

// constructor from "01011" [iop=1] or "02ab"/"02ud" [iop=0] (read from right to left)
onstate::onstate(const string& s, const int iop){
   if(iop == 1){
      _size = s.size();
      _len = (_size-1)/64+1;
      _repr = new unsigned long[_len];
      fill_n(_repr, _len, 0); // initialize, otherwise some unused bits will not be 0.
      for(int i=0; i<_size; i++){
         (*this)[i] = (s[_size-1-i]=='1')? 1 : 0;
      }
   }else if(iop == 0){
      int size = s.size();
       _size = 2*size;
      _len = (_size-1)/64+1;
      _repr = new unsigned long[_len];
      fill_n(_repr, _len, 0); // initialize, otherwise some unused bits will not be 0.
      for(int i=0; i<size; i++){
         if(s[size-1-i] == '0'){
            (*this)[2*i]=0; 
            (*this)[2*i+1]=0;
         }else if(s[size-1-i] == 'a' || s[size-1-i] == 'u'){
            (*this)[2*i]=1; 
            (*this)[2*i+1]=0;
         }else if(s[size-1-i] == 'b' || s[size-1-i] == 'd'){
            (*this)[2*i]=0; 
            (*this)[2*i+1]=1;
         }else if(s[size-1-i] == '2'){
            (*this)[2*i]=1; 
            (*this)[2*i+1]=1;
         }
      } // i
   }else{
      std::cout << "error: no such option for iop=" << iop << std::endl;
      exit(1);
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

// vector<unsigned long> constructors
onstate::onstate(const std::vector<unsigned long> &array, const int n){
   _size = n;
   _len = (n-1)/64+1;
   _repr = new unsigned long[_len];
   copy_n(&array[0], _len, _repr);
};

// unsigned long array constructors
onstate::onstate(const unsigned long *array, const int n){
   _size = n;
   _len = (n-1)/64+1;
   _repr = new unsigned long[_len];
   copy_n(array, _len, _repr);
};

// copy constructor
onstate::onstate(const onstate& state){
   _size = state._size;
   _len  = state._len;
   _repr = new unsigned long[_len];
   copy_n(state._repr, _len, _repr);
}

// copy assignement 
onstate& onstate::operator =(const onstate& state){
   if(this != &state){
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
   _size = state._size;
   _len  = state._len;
   _repr = state._repr;
   state._size = 0;
   state._len  = 0;
   state._repr = nullptr; // to avoid delete the pointed memory
}

// move assignement constructor
onstate& onstate::operator =(onstate&& state){
   if(this != &state){
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

// print 01 string: "010011" (read from right to left)
string onstate::to_string_spinorb() const{
   string s; // loop from left to right for +=
   for(int i = _size-1; i >= 0; i--){
      s += (*this)[i] ? '1' : '0'; 
   }
   return s;
}

// print 2ab0 string
string onstate::to_string(const bool ifcsf) const{
   assert(_size%2 == 0); // only for even no. of basis function 
   string s; 
   int nodd,neven; 
   for(int i = _size/2-1; i >= 0; i--){
      nodd  = (*this)[2*i+1];
      neven = (*this)[2*i];
      if(nodd == 1 && neven == 1)
         s += '2';
      else if(nodd == 0 && neven == 1)
         s += ifcsf? 'u' : 'a';
      else if(nodd == 1 && neven == 0)
         s += ifcsf? 'd' : 'b';
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
void onstate::diff_orb(const onstate& ket,
      vector<int>& cre, 
      vector<int>& ann) const{
   unsigned long idiff,icre,iann;
   // from higher position
   for(int i=_len-1; i>=0; i--){
      idiff = _repr[i] ^ ket._repr[i];
      icre = idiff & _repr[i];
      iann = idiff & ket._repr[i];
#ifdef DEBUG
      for(int j=63; j>=0; j--){
         if(icre & 1ULL<<j){
            cre.push_back(i*64+j);
         }
      }
      for(int j=63; j>=0; j--){
         if(iann & 1ULL<<j){
            ann.push_back(i*64+j);
         }
      }
#else
      while(icre != 0){
         int j = 63-__builtin_clzl(icre);
         cre.push_back(i*64+j);
         icre &= ~(1ULL<<j);
      }
      while(iann != 0){
         int j = 63-__builtin_clzl(iann);
         ann.push_back(i*64+j);
         iann &= ~(1ULL<<j);
      }
#endif
   } // i
}

void onstate::diff_orb(const onstate& ket,
      int* cre, int* ann) const{
   unsigned long idiff,icre,iann;
   int ic=0, ia=0;
   // from higher position
   for(int i=_len-1; i>=0; i--){
      idiff = _repr[i] ^ ket._repr[i];
      icre = idiff & _repr[i];
      iann = idiff & ket._repr[i];
#ifdef DEBUG
      for(int j=63; j>=0; j--){
         if(icre & 1ULL<<j){
            cre[ic] = i*64+j;
            ic++;
         }
      }
      for(int j=63; j>=0; j--){
         if(iann & 1ULL<<j){
            ann[ia] = i*64+j;
            ia++;
         }
      }
#else
      while(icre != 0){
         int j = 63-__builtin_clzl(icre);
         cre[ic] = i*64+j;
         ic++;
         icre &= ~(1ULL<<j);
      }
      while(iann != 0){
         int j = 63-__builtin_clzl(iann);
         ann[ia] = i*64+j;
         ia++;
         iann &= ~(1ULL<<j);
      }
#endif
   } // i
}

// vector version
void onstate::get_olst(std::vector<int>& olst) const{
   olst.clear();
#ifdef DEBUG
   for(int i=0; i<_size; i++){
      if((*this)[i]) olst.push_back(i);
   }
#else
   unsigned long tmp;
   for(int i=0; i<_len; i++){
      tmp = _repr[i];
      while(tmp != 0){
         int j = __builtin_ctzl(tmp);
         olst.push_back(i*64+j);
         tmp &= ~(1ULL<<j);
      }
   }
#endif   
}

// array version
void onstate::get_olst(int* olst) const{
   int ic = 0;
#ifdef DEBUG
   for(int i=0; i<_size; i++){
      if((*this)[i]){
         olst[ic] = i;
         ic++;
      }
   }
#else
   unsigned long tmp;
   for(int i=0; i<_len; i++){
      tmp = _repr[i];
      while(tmp != 0){
         int j = __builtin_ctzl(tmp);
         olst[ic] = i*64+j;
         ic++;
         tmp &= ~(1ULL<<j);
      }
   }
#endif  
}

// vector version
void onstate::get_vlst(std::vector<int>& vlst) const{
   vlst.clear();
#ifdef DEBUG
   for(int i=0; i<_size; i++){
      if(!(*this)[i]) vlst.push_back(i);
   }
#else
   unsigned long tmp;
   for(int i=0; i<_len; i++){
      // be careful about the virtual orbital case 
      tmp = (i!=_len-1)? (~_repr[i]) : ((~_repr[i]) & get_ones((_size%64==0)? 64 : _size%64));
      while(tmp != 0){
         int j = __builtin_ctzl(tmp);
         vlst.push_back(i*64+j);
         tmp &= ~(1ULL<<j);
      }
   }
#endif   
}

// array version
void onstate::get_vlst(int* vlst) const{
   int ic = 0;
#ifdef DEBUG
   for(int i=0; i<_size; i++){
      if(!(*this)[i]){
         vlst[ic] = i;
         ic++;
      }
   }
#else
   unsigned long tmp;
   for(int i=0; i<_len; i++){
      // be careful about the virtual orbital case 
      tmp = (i!=_len-1)? (~_repr[i]) : ((~_repr[i]) & get_ones((_size%64==0)? 64 : _size%64));
      while(tmp != 0){
         int j = __builtin_ctzl(tmp);
         vlst[ic] = i*64+j;
         ic++;
         tmp &= ~(1ULL<<j);
      }
   }
#endif  
}

// perform operations ~(not), ^(xor), &(and) on onstate 

onstate fock::operator ~(const onstate& state1){
   onstate dstate(state1._size);
   for(int i=0; i<state1._len-1; i++){
      dstate._repr[i] = ~state1._repr[i];
   }
   int i = state1._len-1;
   dstate._repr[i] = ((~state1._repr[i]) & get_ones((state1._size%64==0)? 64 : state1._size%64));
   return dstate;
}

onstate fock::operator ^(const onstate& state1, const onstate& state2){
   assert(state1._size == state2._size);
   onstate dstate(state1._size);
   for(int i=0; i<state1._len; i++){
      dstate._repr[i] = state1._repr[i] ^ state2._repr[i];
   }
   return dstate;
}

onstate fock::operator &(const onstate& state1, const onstate& state2){
   assert(state1._size == state2._size);
   onstate dstate(state1._size);
   for(int i=0; i<state1._len; i++){
      dstate._repr[i] = state1._repr[i] & state2._repr[i];
   }
   return dstate;
}
