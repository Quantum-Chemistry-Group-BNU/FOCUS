#ifndef TNS_QSYM_H
#define TNS_QSYM_H

#include "../core/serialization.h"
#include "../core/onspace.h"
#include <string>
#include <map>

namespace tns{

// qsym = (Ne,Na)
class qsym{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & _ne;
	 ar & _na;
      }
   public:
      // constructors
      qsym(): _ne(0), _na(0) {};
      qsym(const int ne, const int na): _ne(ne), _na(na) {};
      // assignement
      qsym& operator =(const qsym& sym){
	 _ne = sym._ne;
	 _na = sym._na;
	 return *this;
      }
      // to be used as key in map: lexicographical order
      bool operator <(const qsym& sym) const{
	 return (_ne < sym._ne) || (_ne == sym._ne && _na < sym._na); 
      }
      bool operator ==(const qsym& sym) const{
	 return (_ne == sym._ne) && (_na == sym._na); 
      }
      bool operator !=(const qsym& sym) const{
	 return !(*this == sym);
      }
      inline int ne() const{ return _ne; }
      inline int na() const{ return _na; }
      inline double parity() const{ return -2*(_ne%2)+1; }
      // print
      std::string to_string() const;
      friend std::ostream& operator <<(std::ostream& os, const qsym& sym);
      // Abelian symmetry
      qsym operator -() const{ return qsym(-_ne,-_na); }
      qsym& operator +=(const qsym& sym1){ 
	 _ne += sym1._ne;
	 _na += sym1._na;
	return *this; 
      }
      friend qsym operator +(const qsym& sym1, const qsym& sym2);
      friend qsym operator -(const qsym& sym1, const qsym& sym2);
   private:
      int _ne, _na;
};

// qsym_qspace
using qsym_space = std::map<qsym,int>;
int qsym_space_dim(const qsym_space& qs);
void qsym_space_print(const qsym_space& qs, const std::string& name);

// direct product table of qsym_space : V1*V2->V12
using qsym_dpt = std::map<qsym,std::map<std::pair<qsym,qsym>,std::tuple<int,int,int>>>;
std::pair<qsym_space,qsym_dpt> qsym_space_dpt(const qsym_space& qs1, 
					      const qsym_space& qs2);

// physical degree of freedoms
extern const std::vector<qsym> phys_sym;
extern const fock::onspace phys_space;
extern const qsym_space phys_qsym_space;

// vacuum
extern const qsym_space vac_qsym_space;

} // tns

#endif
