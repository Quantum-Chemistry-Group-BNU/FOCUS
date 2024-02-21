#ifndef CSF_H
#define CSF_H

#include "binom.h"
#include "onstate.h"

namespace fock{

   struct csfstate{
      public:
         // constructor
         csfstate(const int k){ 
            onstate _repr(2*k);
            repr = _repr;  
         }
         // no. of deltaS=1/2
         int nelec() const{ return repr.nelec(); };
         int twos() const{ return repr.nelec_a()-repr.nelec_b(); };
      public:
         onstate repr; // internal representation
   };

   // --- CSF space ---
   using csfspace = std::vector<csfstate>;

   // csf space dimsion by Weyl-Paldus formula
   inline size_t dim_csf_space(const int k, const int n, const int ts){
      size_t dim = (n%2!=ts%2)? 0 : (ts+1)*fock::binom(k+1,(n-ts)/2)*fock::binom(k+1,(n+ts)/2+1)/(k+1);
      return dim;
   }

   // generate all CSF with (N,S) via a iterative procedure
   csfspace get_csf_space(const int k, const int n, const int ts);

} // fock

#endif
