#ifndef BINOM_H
#define BINOM_H

#include <cstddef> // size_t

namespace fock{

   // https://stackoverflow.com/questions/44718971/calculate-binomial-coffeficient-very-reliably
   constexpr inline size_t binom(size_t n, size_t k) noexcept
   {
      return
         (        k> n  )? 0 :          // out of range
         (k==0 || k==n  )? 1 :          // edge
         (k==1 || k==n-1)? n :          // first
         (     k+k < n  )?              // recursive:
         (binom(n-1,k-1) * n)/k :       //  path to k=1   is faster
         (binom(n-1,k) * n)/(n-k);      //  path to k=n-1 is faster
   }

} // fock

#endif
