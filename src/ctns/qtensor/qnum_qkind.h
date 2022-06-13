#ifndef QNUM_QKIND_H
#define QNUM_QKIND_H

#include <complex>
#include <string>

namespace ctns{

// kind of CTNS
namespace qkind{ 

// isym = 0:
struct rZ2{
   using dtype = double;
   static const int isym = 0;
   static const bool ifkr = false;
};
struct cZ2{
   using dtype = std::complex<double>;
   static const int isym = 0;
   static const bool ifkr = false;
};
// isym = 1:
struct rN{
   using dtype = double;
   static const int isym = 1;
   static const bool ifkr = false;
};
struct cN{
   using dtype = std::complex<double>;
   static const int isym = 1;
   static const bool ifkr = false;
};
// isym = 2:
struct rNSz{
   using dtype = double;
   static const int isym = 2;
   static const bool ifkr = false;
};
struct cNSz{
   using dtype = std::complex<double>;
   static const int isym = 2;
   static const bool ifkr = false;
};
// Kramers symmetry: relativistic H with SOC
struct cNK{
   using dtype = std::complex<double>;
   static const int isym = 1;
   static const bool ifkr = true;
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

// get_name
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
