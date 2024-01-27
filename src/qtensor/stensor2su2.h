#ifndef STENSOR2SU2_H
#define STENSOR2SU2_H

#include "qtensor2.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void qtensor2<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor2: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc;
         for(int i=0; i<info._nnzaddr.size(); i++){
            auto key = info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key); 
            const auto blk = (*this)(br,bc);
            if(level >= 1){
               std::cout << "i=" << i << " block["  
                  << br << ":" << info.qrow.get_sym(br) << "," 
                  << bc << ":" << info.qcol.get_sym(bc) << "]" 
                  << " dim0,dim1=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(i));
            } // level>=1
         } // i
      }

} // ctns

#endif
