#ifndef CTNS_KIND_H
#define CTNS_KIND_H

#include <complex>
#include "ctns_qsym.h"

namespace ctns{

// kind of CTNS
namespace kind{ 

// is_available
template <typename Tm>
bool is_available(){ return false; }

// (N): real & complex 
struct rN{
   using dtype = double;
   static const int isym = 1;
   static const bool is_kramers = false;
};
template <>
bool is_available<rN>(){ return true; } 

struct cN{
   using dtype = std::complex<double>;
   static const int isym = 1;
   static const bool is_kramers = false;
};
template <>
bool is_available<cN>(){ return true; } 

// (N,Sz): real & complex
struct rNSz{
   using dtype = double;
   static const int isym = 2;
   static const bool is_kramers = false;
};
template <>
bool is_available<rNSz>(){ return true; }

struct cNSz{
   using dtype = std::complex<double>;
   static const int isym = 2;
   static const bool is_kramers = false;
};
template <>
bool is_available<cNSz>(){ return true; }

// (N,K): complex -> relativistic H with SOC
struct cNK{
   using dtype = std::complex<double>;
   static const int isym = 1;
   static const bool is_kramers = true;
};
template <>
bool is_available<cNK>(){ return true; }

} // kind

} // ctns

#endif
