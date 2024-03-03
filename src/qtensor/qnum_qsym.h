#ifndef QNUM_QSYM_H
#define QNUM_QSYM_H

#include "../core/serialization.h"
#include "../core/onstate.h"
#include <string>

namespace ctns{

   //
   // Quantum number class for Abelian symmetry: 
   //
   //  isym = 0: Z2
   //  isym = 1: N=(Ne,0)
   //  isym = 2: NSz=(Ne,2M)
   //  isym = 3: NS=(Ne,S) [tm is ts in this case!]
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
         qsym(const short isym, const short ne): _isym(isym), _ne(ne), _tm(0) {} 
         qsym(const short isym, const short ne, const short tm): _isym(isym), _ne(ne), _tm(tm) {
            // ne,tm can be negative due to the negative function 
            if( (isym==2 or isym==3) && (std::abs(ne)%2 != std::abs(tm)%2) ){ 
               std::cout << "error: inconsistent (ne,tm) for isym=" << isym 
                  << " (ne,tm)=" << ne << "," << tm << std::endl;
               exit(1); 
            }
         }
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
         short ts() const{ 
            assert(_isym==3); 
            return _tm; 
         }
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
            assert(_isym != 3); 
            if(_isym == 0){
               return qsym(_isym, _ne, 0);
            }else{
               return qsym(_isym, -_ne, -_tm); 
            }
         }
      private:
         short _isym = 0;
         short _ne = 0; // na+nb
         short _tm = 0; // na-nb or twoS
   };

   // get qsym for a given onstate
   inline qsym get_qsym_onstate(const short isym, const fock::onstate& state){
      short ne = (isym==0)? state.nelec()%2 : state.nelec();
      short tm = (isym<=1)? 0 : state.twoms();
      return qsym(isym, ne, tm);
   }

   // set qsym for state
   inline qsym get_qsym_state(const short isym, const int nelec, const int twoms){
      qsym sym_state;
      if(isym == 0){
         sym_state = qsym(isym, nelec%2, 0);
      }else if(isym == 1){
         sym_state = qsym(isym, nelec, 0);
      }else if(isym == 2){
         sym_state = qsym(isym, nelec, twoms);
      }
      return sym_state;
   }

   // kA+/kB+
   inline qsym get_qsym_opC(const short isym, const int p){
      int spin = p%2;
      qsym sym_op;
      if(isym == 0 or isym == 1){
         sym_op = qsym(isym,1,0);
      }else if(isym == 2){ 
         sym_op = (spin==0)? qsym(isym,1,1) : qsym(isym,1,-1);
      }
      return sym_op;
   }
   // Apq = p^+q^+
   inline qsym get_qsym_opA(const short isym, const int p, const int q){
      int spin1 = p%2, spin2 = q%2;
      qsym sym_op;
      if(isym == 0){
         sym_op = qsym(isym,0,0);
      }else if(isym == 1){
         sym_op = qsym(isym,2,0);
      }else if(isym == 2){
         if(spin1 != spin2){
            sym_op = qsym(isym,2,0);
         }else{
            sym_op = (spin1==0)? qsym(isym,2,2) : qsym(isym,2,-2);
         }
      }
      return sym_op;
   }
   // Bps = p^+s
   inline qsym get_qsym_opB(const short isym, const int p, const int s){
      int spin1 = p%2, spin2 = s%2;
      qsym sym_op;
      if(isym == 0 or isym == 1){
         sym_op = qsym(isym,0,0);
      }else if(isym == 2){
         if(spin1 == spin2){
            sym_op = qsym(isym,0,0);
         }else{
            sym_op = (spin1==0)? qsym(isym,0,2) : qsym(isym,0,-2);
         }
      }
      return sym_op;
   }
   // Sp: qsym of ap^+Sp must be zero
   inline qsym get_qsym_opS(const short isym, const int p){
      return -get_qsym_opC(isym, p);
   }
   // Ppq: qsym of ApqPpq must be zero
   inline qsym get_qsym_opP(const short isym, const int p, const int q){
      return -get_qsym_opA(isym, p, q);
   }
   // Qps: qsym of BpsQps must be zero
   inline qsym get_qsym_opQ(const short isym, const int p, const int s){
      return -get_qsym_opB(isym, p, s);
   }

} // ctns

#endif
