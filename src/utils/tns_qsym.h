#ifndef TNS_QSYM_H
#define TNS_QSYM_H

#include "../core/serialization.h"
#include "../core/onspace.h"
#include <string>
#include <map>

namespace tns{

// --- qsym = (Ne,Na) ---
class qsym{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & _ne;
	 ar & _na;
      }
   public:
      qsym(): _ne(0), _na(0) {};
      qsym(const int ne, const int na): _ne(ne), _na(na) {};
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
      qsym operator -() const{ return qsym(-_ne,-_na); }
      std::string to_string() const;
      friend std::ostream& operator <<(std::ostream& os, const qsym& sym);
      // Abelian symmetry
      friend qsym operator +(const qsym& sym1, const qsym& sym2);
      friend qsym operator -(const qsym& sym1, const qsym& sym2);
   private:
      int _ne, _na;
};

// --- qspace ---
using qsym_space = std::map<qsym,int>;
int qsym_space_dim(const qsym_space& qs);
void qsym_space_print(const qsym_space& qs, const std::string& name);

// --- physical degree of freedoms  ---
extern const std::vector<qsym> phys_sym;
extern const fock::onspace phys_space;
extern const qsym_space phys_qsym_space;

} // tns

#endif
