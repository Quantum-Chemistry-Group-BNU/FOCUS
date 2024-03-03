#ifndef STENSOR3SU2_H
#define STENSOR3SU2_H

#include "qtensor3.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void qtensor3<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor3: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc, bm, tsi;
         for(int i=0; i<info._nnzaddr.size(); i++){
            auto key = info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            const auto blk = (*this)(br,bc,bm,tsi);
            if(level >= 1){
               std::cout << "i=" << i << " block["
                  << br << ":" << info.qrow.get_sym(br) << "," 
                  << bc << ":" << info.qcol.get_sym(bc) << "," 
                  << bm << ":" << info.qmid.get_sym(bm) << ";"
                  << tsi << "]" 
                  << " dim0,dim1,dim2=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ","
                  << blk.dim2 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(i));
            } // level>=1
         } // i
      }

   // get block by sym used in rcanon_CIcoeff
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      dtensor3<Tm> qtensor3<ifab,Tm>::get_rcf_symblk(const qsym qr, const qsym qc, const qsym qm) const{
         int br = info.qrow.existQ(qr);
         int bc = info.qcol.existQ(qc);
         int bm = info.qmid.existQ(qm);
         if(br == -1 || bc == -1 || bm == -1){
            return dtensor3<Tm>();
         }
         // CRcouple
         assert(info.couple == CRcouple);
         int tsi = info.qrow.get_sym(br).ts(); 
         return (*this)(br,bc,bm,tsi);
      }

} // ctns

#endif
