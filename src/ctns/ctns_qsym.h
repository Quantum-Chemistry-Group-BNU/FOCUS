#ifndef CTNS_QSYM_H
#define CTNS_QSYM_H

#include "../core/onstate.h" // qsym of det
#include "../core/tools.h"
#include "../core/serialization.h"
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
      int _ne, _tm; // na+nb, na-nb
};

// onstate
template <typename Tm>
qsym get_qsym(const fock::onstate& state){
   int ne = state.nelec();
   int tm = tools::is_complex<Tm>()? 0 : state.twoms();
   return qsym(ne,tm);
}

// qsym_qspace
class qsym_space{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dims;
      }
   public:
      // constructor
      qsym_space(){}
      qsym_space(std::vector<std::pair<qsym,int>> ds): dims(ds) {}
      // helpers
      inline int size() const{ return dims.size(); }
      inline qsym get_sym(const int i) const{ return dims[i].first; } 
      inline int get_dim(const int i) const{ return dims[i].second; }
      // total dimension
      int get_dimAll() const{
         int dim = 0;
         for(const auto& p : dims) dim += p.second;
         return dim;
      }
      // offset 
      std::vector<int> get_offset() const{
         std::vector<int> offset;
         int ioff = 0;
	 for(int i=0; i<dims.size(); i++){
	    offset.push_back(ioff);
	    ioff += dims[i].second; 
	 }
	 return offset;
      }
      // comparison
      bool operator ==(const qsym_space& qs) const{
	 bool ifeq = dims.size() == qs.size();
	 if(not ifeq) return false;
	 for(int i=0; i<dims.size(); i++){
	    ifeq = ifeq && dims[i].first == qs.dims[i].first &&
		           dims[i].second == qs.dims[i].second;
	    if(not ifeq) return false;
	 }
	 return true;
      }
      void print(const std::string name) const{
	 std::cout << "qsym_space: " << name << " nsym=" << dims.size() 
      	           << " dimAll=" << get_dimAll() << std::endl;
         // loop over symmetry sectors
         for(int i=0; i<dims.size(); i++){
            auto sym = dims[i].first;
            auto dim = dims[i].second;
	    std::cout << " " << sym << ":" << dim;
         }
	 std::cout << std::endl;
      }
   public:
      std::vector<std::pair<qsym,int>> dims;
};

/*
// direct product table of qsym_space : V1*V2->V12
using qsym_dpt = std::map<qsym,std::map<std::pair<qsym,qsym>,std::tuple<int,int,int>>>;
std::pair<qsym_space,qsym_dpt> qsym_space_dpt(const qsym_space& qs1, 
					      const qsym_space& qs2);
*/

} // tns

#endif
