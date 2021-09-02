#ifndef CTNS_QSYM_H
#define CTNS_QSYM_H

#include "../../core/serialization.h"
#include "../../core/onstate.h"
#include <string>

namespace ctns{

// Quantum number class for Abelian symmetry: 
//
//  isym = 0: Z2
//  isym = 1: N=(Ne,0)
//  isym = 2: NSz=(Ne,2M)
//
class qsym{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & _isym;
	 ar & _ne;
	 ar & _tm;
      }
   public:
      // constructors
      qsym(){}
      qsym(const short isym): _isym(isym), _ne(0), _tm(0) {} 
      qsym(const short isym, const short ne, const short tm):  _isym(isym), _ne(ne), _tm(tm) {}
      // assignement
      qsym& operator =(const qsym& sym){
         _isym = sym._isym; // allow _isym to be changed here
	 _ne = sym._ne;
	 _tm = sym._tm;
	 return *this;
      }
      // to be used as key in map: lexicographical order
      bool operator <(const qsym& sym) const{
         assert(_isym == sym._isym);
	 return (_ne < sym._ne) || (_ne == sym._ne && _tm < sym._tm); 
      }
      bool operator ==(const qsym& sym) const{
         assert(_isym == sym._isym);
	 return (_ne == sym._ne) && (_tm == sym._tm); 
      }
      bool operator !=(const qsym& sym) const{
         assert(_isym == sym._isym);
	 return !(*this == sym);
      }
      bool is_zero() const{ return _ne==0 && _tm==0; }
      bool is_nonzero() const{ return _ne!=0 || _tm!=0; }
      short isym() const{ return _isym; }
      short ne() const{ return _ne; }
      short tm() const{ return _tm; }
      short parity() const{ return _ne%2; }
      // print
      std::string to_string() const{ 
         if(_isym <= 1){
            return "("+std::to_string(_ne)+")";
         }else{ 
            return "("+std::to_string(_ne)+","+std::to_string(_tm)+")";
	 }
      }
      friend std::ostream& operator <<(std::ostream& os, const qsym& sym){
         os << sym.to_string();
         return os;
      }
      // Abelian symmetry
      qsym& operator +=(const qsym& sym1){
         assert(_isym == sym1._isym); 
         if(_isym == 0){ // Z2 symmetry
	    _ne = (_ne + sym1._ne)%2;
         }else{
            _ne += sym1._ne;
	    _tm += sym1._tm;
	 }	 
	 return *this; 
      }
      friend qsym operator +(const qsym& sym1, const qsym& sym2){
         assert(sym1._isym == sym2._isym);
	 if(sym1._isym == 0){
            return qsym(sym1._isym, (sym1._ne + sym2._ne)%2, 0);
	 }else{
            return qsym(sym1._isym, sym1._ne + sym2._ne, sym1._tm + sym2._tm);
	 }
      }
      friend qsym operator -(const qsym& sym1, const qsym& sym2){
         auto sym2i = -sym2;
         return sym1 + sym2i;
      }
      qsym flip() const{ return qsym(_isym, _ne, -_tm); }
      qsym operator -() const{ 
	 if(_isym == 0){
            return qsym(_isym, _ne, 0);
	 }else{
            return qsym(_isym, -_ne, -_tm); 
	 }
      }
   private:
      short _isym = 0;
      short _ne = 0; // na+nb
      short _tm = 0; // na-nb
};

// get qsym for a given onstate
inline qsym get_qsym_onstate(const short isym, const fock::onstate& state){
   short ne = (isym==0)? state.nelec()%2 : state.nelec();
   short tm = (isym<=1)? 0 : state.twoms();
   return qsym(isym, ne, tm);
}

} // ctns

#endif
