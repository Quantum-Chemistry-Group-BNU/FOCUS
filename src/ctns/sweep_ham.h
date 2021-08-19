#ifndef SWEEP_HAM_H
#define SWEEP_HAM_H

#include <functional> // for std::function

namespace ctns{

template <typename Tm>
using HVec_type = std::function<void(Tm*, const Tm*)>;

template <typename Tm>
struct Hx_functor{
   public:
      // compute
      qtensor3<Tm> operator ()(){ return opxwf(); }
   public:
      std::function<qtensor3<Tm>()> opxwf;
};
template <typename Tm>
using Hx_functors = std::vector<Hx_functor<Tm>>;

} // ctns

#endif
