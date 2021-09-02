#ifndef CTNS_QKIND_H
#define CTNS_QKIND_H

#include <complex>
#include <string>

namespace ctns{

// kind of CTNS
namespace qkind{ 

// isym = 0:
struct rZ2{
   static const int isym = 0;
   using dtype = double;
};
struct cZ2{
   static const int isym = 0;
   using dtype = std::complex<double>;
};
// isym = 1:
struct rN{
   static const int isym = 1;
   using dtype = double;
};
struct cN{
   static const int isym = 1;
   using dtype = std::complex<double>;
};
// isym = 2:
struct rNSz{
   static const int isym = 2;
   using dtype = double;
};
struct cNSz{
   static const int isym = 2;
   using dtype = std::complex<double>;
};
// Kramers symmetry: relativistic H with SOC
struct cNK{
   static const int isym = 1;
   using dtype = std::complex<double>;
};

// is_available
template <typename Tm>
inline bool is_available(){ return false; }
template <> inline bool is_available<rZ2>(){ return true; } 
template <> inline bool is_available<cZ2>(){ return true; } 
template <> inline bool is_available<rN>(){ return true; } 
template <> inline bool is_available<cN>(){ return true; } 
template <> inline bool is_available<rNSz>(){ return true; }
template <> inline bool is_available<cNSz>(){ return true; }
template <> inline bool is_available<cNK>(){ return true; }

template <typename>
inline bool is_kramers(){ return false; }
template <> inline bool is_kramers<cNK>(){ return true; }

template <typename Tm>
inline std::string get_name(){ return ""; }
template <> inline std::string get_name<rZ2>(){ return "rZ2"; } 
template <> inline std::string get_name<cZ2>(){ return "cZ2"; } 
template <> inline std::string get_name<rN>(){ return "rN"; } 
template <> inline std::string get_name<cN>(){ return "cN"; } 
template <> inline std::string get_name<rNSz>(){ return "rNSz"; }
template <> inline std::string get_name<cNSz>(){ return "cNSz"; }
template <> inline std::string get_name<cNK>(){ return "cNK"; }

} // qkind

} // ctns

#endif
