#ifndef SPIN_H
#define SPIN_H

#include <gsl/gsl_sf_coupling.h>

namespace fock{

   inline bool spin_triangle(const int ts1, const int ts2, const int ts3){
      return (ts1+ts2-ts3)%2==0 && std::abs(ts1-ts2) <= ts3 && ts3 <= ts1+ts2;
   }

   // <ja,ma,jb,mb|jc,mc>
   inline double cgcoeff(const int two_ja, 
         const int two_jb, 
         const int two_jc, 
         const int two_ma, 
         const int two_mb, 
         const int two_mc){
      int sum = (-two_ja+two_jb-two_mc)/2;
      return ::gsl_sf_coupling_3j(two_ja,two_jb,two_jc,
            two_ma,two_mb,-two_mc)*std::sqrt(two_jc+1.0)*std::pow(-1,sum);
   }

   inline double wigner3j(const int two_ja, 
         const int two_jb, 
         const int two_jc, 
         const int two_ma, 
         const int two_mb, 
         const int two_mc){
      return ::gsl_sf_coupling_3j(two_ja,two_jb,two_jc,
            two_ma,two_mb,two_mc);
   }

   inline double wigner6j(const int two_ja, 
         const int two_jb, 
         const int two_jc, 
         const int two_jd, 
         const int two_je, 
         const int two_jf){
      return ::gsl_sf_coupling_6j(two_ja,two_jb,two_jc,
            two_jd,two_je,two_jf);
   }

   inline double wigner9j(const int two_ja, 
         const int two_jb, 
         const int two_jc, 
         const int two_jd, 
         const int two_je, 
         const int two_jf,
         const int two_jg, 
         const int two_jh, 
         const int two_ji){
      return ::gsl_sf_coupling_9j(two_ja,two_jb,two_jc,
            two_jd,two_je,two_jf,two_jg,two_jh,two_ji);
   }

} // fock

#endif
