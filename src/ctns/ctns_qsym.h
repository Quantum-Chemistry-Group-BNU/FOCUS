#ifndef CTNS_QSYM_H
#define CTNS_QSYM_H

#include "../core/serialization.h"
#include "../core/onspace.h"
#include <string>
#include <map>

namespace ctns{

// Abelian symmetry: (Ne,2M)
// Htype=1: setting 2M=0 for all qsym
class qsym{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & _ne;
	 ar & _tm;
      }
   public:
      // constructors
      qsym(): _ne(0), _tm(0) {};
      qsym(const int ne, const int tm): _ne(ne), _tm(tm) {};
      // assignement
      qsym& operator =(const qsym& sym){
	 _ne = sym._ne;
	 _tm = sym._tm;
	 return *this;
      }
      // to be used as key in map: lexicographical order
      bool operator <(const qsym& sym) const{
	 return (_ne < sym._ne) || (_ne == sym._ne && _tm < sym._tm); 
      }
      bool operator ==(const qsym& sym) const{
	 return (_ne == sym._ne) && (_tm == sym._tm); 
      }
      bool operator !=(const qsym& sym) const{
	 return !(*this == sym);
      }
      inline int ne() const{ return _ne; }
      inline int tm() const{ return _tm; }
      inline int parity() const{ return _ne%2; }
      // print
      std::string to_string() const;
      friend std::ostream& operator <<(std::ostream& os, const qsym& sym);
      // Abelian symmetry
      qsym flip() const{ return qsym(_ne,-_tm); }
      qsym operator -() const{ return qsym(-_ne,-_tm); }
      qsym& operator +=(const qsym& sym1){ 
	 _ne += sym1._ne;
	 _tm += sym1._tm;
	return *this; 
      }
      friend qsym operator +(const qsym& sym1, const qsym& sym2);
      friend qsym operator -(const qsym& sym1, const qsym& sym2);
   private:
      int _ne, _tm;
};

// qsym_qspace
using qsym_space = std::map<qsym,int>;
int qsym_space_dim(const qsym_space& qs);
void qsym_space_print(const qsym_space& qs, const std::string& name);
// vacuum
extern const qsym_space vac_qsym_space;
// physical degree of freedoms (depending on Htype)
extern const fock::onspace phys_space;
std::vector<qsym> get_phys_sym(const bool Htype);
qsym_space get_phys_qsym_space(const bool Htype);

// direct product table of qsym_space : V1*V2->V12
using qsym_dpt = std::map<qsym,std::map<std::pair<qsym,qsym>,std::tuple<int,int,int>>>;
std::pair<qsym_space,qsym_dpt> qsym_space_dpt(const qsym_space& qs1, 
					      const qsym_space& qs2);

} // tns

#endif
