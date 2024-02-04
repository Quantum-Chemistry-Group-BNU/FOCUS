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

   // ZL20200531: Permute the line of diagrams, while maintaining their directions
   // 	       This does not change the tensor, but just permute order of index
   //         
   //           i --<--*--<-- j => j -->--*-->-- i
   //
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::P() const{
         qtensor2<ifab,Tm> qt2(info.sym, info.qcol, info.qrow, 
               {std::get<1>(info.dir), std::get<0>(info.dir)});
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            br = std::get<0>(key); 
            bc = std::get<1>(key); 
            auto blk = qt2(br,bc);
            // transpose
            const auto blkt = (*this)(bc,br);
            for(int ic=0; ic<blk.dim1; ic++){
               for(int ir=0; ir<blk.dim0; ir++){
                  blk(ir,ic) = blkt(ic,ir);
               } // ir
            } // ic
         } // i
         return qt2; 
      }

   // return the adjoint tensor
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::H() const{
         // symmetry of operator get changed in consistency with line changes
         qsym sym(3,-info.sym.ne(),info.sym.ts());
         qtensor2<ifab,Tm> qt2(sym, info.qcol, info.qrow, info.dir);
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            auto blk = qt2(br,bc);
            // conjugate transpose
            const auto blkh = (*this)(bc,br);
            for(int ic=0; ic<blk.dim1; ic++){
               for(int ir=0; ir<blk.dim0; ir++){
                  blk(ir,ic) = tools::conjugate(blkh(ic,ir));
               } // ir
            } // ic
           
            // determine the prefactor for Adjoint tensor operator 
            auto symr = qt2.info.qrow.get_sym(br);
            auto symc = qt2.info.qcol.get_sym(bc);
            int tsr = symr.ts();
            int tsc = symc.ts();
            int deltats = info.sym.ts() + tsr - tsc;
            Tm fac = deltats%2==0? 1.0 : -1.0;
            fac *= std::sqrt((2.0*tsc+1.0)/(2.0*tsr+1.0));
            linalg::xscal(blk.size(), fac, blk.data());

         } // i
         return qt2; 
      }

} // ctns

#endif
