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
   template <typename Tm>
      void HermitianConjugate(const stensor2su2<Tm>& qt1,
            stensor2su2<Tm>& qt2,
            const bool adjoint){
         assert(qt1.size() == qt2.size());
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            auto blk = qt2(br,bc);
            // conjugate transpose
            const auto blkh = qt1(bc,br);
            for(int ic=0; ic<blk.dim1; ic++){
               for(int ir=0; ir<blk.dim0; ir++){
                  blk(ir,ic) = tools::conjugate(blkh(ic,ir));
               } // ir
            } // ic
              //-----------------------------------------------------
              // determine the prefactor for Adjoint tensor operator 
            if(adjoint){
               // <br||Tk_bar||bc> = (-1)^{k-jc+jr}sqrt{[jc]/[jr]}<bc||Tk||br>*
               auto symr = qt2.info.qrow.get_sym(br);
               auto symc = qt2.info.qcol.get_sym(bc);
               int tsr = symr.ts();
               int tsc = symc.ts();
               int deltats = (qt2.info.sym.ts() + tsr - tsc);
               assert(deltats%2 == 0);
               Tm fac = (deltats/2)%2==0? 1.0 : -1.0;
               fac *= std::sqrt((tsc+1.0)/(tsr+1.0));
               linalg::xscal(blk.size(), fac, blk.data());
            }
            //----------------------------------------------------
         } // i
      }
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::H(const bool adjoint) const{
         // symmetry of operator get changed in consistency with line changes
         qsym sym(3,-info.sym.ne(),info.sym.ts());
         qtensor2<ifab,Tm> qt2(sym, info.qcol, info.qrow, info.dir);
         HermitianConjugate(*this, qt2, adjoint);
         return qt2; 
      }

   // align
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::align_qrow(const qbond& qrow2) const{
         assert(info.qrow.sort_by_sym() == qrow2.sort_by_sym());
         if(info.qrow == qrow2) return *this;
         // align
         qtensor2<ifab,Tm> qt2(info.sym, qrow2, info.qcol, info.dir);
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            int br = std::get<0>(key);
            int bc = std::get<1>(key);
            auto blk = qt2(br,bc);
            // find the location: work for small qrow2
            int br0 = info.qrow.existQ(qrow2.get_sym(br));
            const auto blk0 = (*this)(br0,bc);
            assert(blk0.size() == blk.size());
            linalg::xcopy(blk0.size(), blk0.data(), blk.data());
         }
         return qt2;
      }

} // ctns

#endif
