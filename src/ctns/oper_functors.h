#ifndef OPER_FUNCTORS_H
#define OPER_FUNCTORS_H

#include <functional> // for std::function

namespace ctns{

template <typename Tm>
struct Hx_functor{
   public:
      // constructor
      Hx_functor(const std::string _label, 
		 const int _index=0,
		 const int _iformula=0){
	 label = _label;
         index = _index;
         iformula = _iformula;
      }
      // compute
      stensor3<Tm> operator ()() const{ return opxwf(); }
      // print
      friend std::ostream& operator <<(std::ostream& os, const Hx_functor& Hx){
         os << " Hx_functor: label=" << Hx.label 
	    << " index=" << Hx.index
	    << " iformula=" << Hx.iformula; 
         return os;
      }
   public:
      std::string label;
      int index, iformula;
      std::function<stensor3<Tm>()> opxwf;
};
template <typename Tm>
using Hx_functors = std::vector<Hx_functor<Tm>>;

// for Davidson algorithm
template <typename Tm>
using HVec_type = std::function<void(Tm*, const Tm*)>;

} // ctns

#endif
