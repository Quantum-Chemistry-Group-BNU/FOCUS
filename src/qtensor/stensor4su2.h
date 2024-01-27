#ifndef STENSOR4SU2_H
#define STENSOR4SU2_H

#include "qtensor4.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void qtensor4<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor4: " << name << " own=" << own << " _data=" << _data << std::endl;
         info.print(name);
         int br, bc, bm, bv, tsi, tsj;
         for(int i=0; i<info._nnzaddr.size(); i++){
            auto key = info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            bv = std::get<3>(key);
            tsi = std::get<4>(key);
            tsj = std::get<5>(key);
            const auto blk = (*this)(br,bc,bm,bv,tsi,tsj);
            if(level >= 1){
               std::cout << "i=" << i << " block[" 
                  << br << ":" << info.qrow.get_sym(br) << "," 
                  << bc << ":" << info.qcol.get_sym(bc) << ","
                  << bm << ":" << info.qmid.get_sym(bm) << "," 
                  << bv << ":" << info.qver.get_sym(bv) << ";"
                  << tsi << "," << tsj << "]" 
                  << " dim0,dim1,dim2,dim3=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << "," 
                  << blk.dim2 << "," 
                  << blk.dim3 << ")"
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(i));
            } // level>=1
         } // i
      }

} // ctns

#endif
