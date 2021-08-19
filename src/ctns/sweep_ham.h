#ifndef SWEEP_HAM_H
#define SWEEP_HAM_H

#include <functional> // for std::function
#include "oper_dict.h" 

namespace ctns{

template <typename Tm>
using HVec_type = std::function<void(Tm*, const Tm*)>;

template <typename Tm>
struct Hx_functor{
   public:
      // constructor
      Hx_functor(const std::string _label, const int _iformula, const int _index){
	 label = _label;
         iformula = _iformula;
         index = _index;
      }
      // print
      friend std::ostream& operator <<(std::ostream& os, const Hx_functor& Hx){
         os << " label=" << Hx.label << " iformula=" << Hx.iformula << " index=";
         if(Hx.label == "AP" || Hx.label == "PA" || Hx.label == "BQ" || Hx.label == "QB"){
            auto pq = oper_unpack(Hx.index);
            os << pq.first << " " << pq.second;
         }else{
            os << Hx.index;
         }
         return os;
      }
      // compute
      qtensor3<Tm> operator ()(){ return opxwf(); }
   public:
      std::string label;
      int iformula, index;
      std::function<qtensor3<Tm>()> opxwf;
};
template <typename Tm>
using Hx_functors = std::vector<Hx_functor<Tm>>;

} // ctns

#endif
