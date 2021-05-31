#ifndef CTNS_QSYM_H
#define CTNS_QSYM_H

#include "../../core/serialization.h"
#include "../../core/onstate.h"
#include <string>

namespace ctns{

// Quantum number class for Abelian symmetry: NSz=(Ne,2M); N=(Ne,0)
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
      qsym(const int ne): _ne(ne), _tm(0) {};
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
      std::string to_string() const{ return "("+std::to_string(_ne)+","+std::to_string(_tm)+")"; }
      friend std::ostream& operator <<(std::ostream& os, const qsym& sym){
         os << sym.to_string();
         return os;
      }
      // Abelian symmetry
      qsym flip() const{ return qsym(_ne,-_tm); }
      qsym operator -() const{ return qsym(-_ne,-_tm); }
      qsym& operator +=(const qsym& sym1){ 
	 _ne += sym1._ne;
	 _tm += sym1._tm;
	return *this; 
      }
      friend qsym operator +(const qsym& sym1, const qsym& sym2){
         return qsym(sym1._ne + sym2._ne, sym1._tm + sym2._tm);
      }
      friend qsym operator -(const qsym& sym1, const qsym& sym2){
         return qsym(sym1._ne - sym2._ne, sym1._tm - sym2._tm);
      }
   private:
      int _ne, _tm; // na+nb, na-nb
};

// get qsym for a given onstate
inline qsym get_qsym_onstate(const int isym, const fock::onstate& state){
   int ne = state.nelec();
   int tm = (isym==1)? 0 : state.twoms();
   return qsym(ne,tm);
}

// qsym_qspace for bond of CTNS: std::vector<std::pair<qsym,int>> dims
class qbond{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dims;
      }
   public:
      // constructor
      qbond(){}
      qbond(const std::vector<std::pair<qsym,int>>& ds): dims(ds) {}
      // helpers
      inline int size() const{ return dims.size(); }
      inline qsym get_sym(const int i) const{ return dims[i].first; } 
      inline int get_dim(const int i) const{ return dims[i].second; }
      inline int get_parity(const int i) const{ return dims[i].first.parity(); }
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
      bool operator ==(const qbond& qs) const{
	 bool ifeq = dims.size() == qs.size();
	 if(not ifeq) return false;
	 for(int i=0; i<dims.size(); i++){
	    ifeq = ifeq && dims[i].first == qs.dims[i].first &&
		           dims[i].second == qs.dims[i].second;
	    if(not ifeq) return false;
	 }
	 return true;
      }
      void print(const std::string name, const bool debug=true) const{
	 std::cout << " qbond: " << name << " nsym=" << dims.size()
      	           << " dimAll=" << get_dimAll() << std::endl;
         // loop over symmetry sectors
	 if(debug){
            for(int i=0; i<dims.size(); i++){
               auto sym = dims[i].first;
               auto dim = dims[i].second;
	       std::cout << " " << sym << ":" << dim;
            }
	    std::cout << std::endl;
	 }
      }
   public:
      std::vector<std::pair<qsym,int>> dims;
};

} // ctns

#endif
