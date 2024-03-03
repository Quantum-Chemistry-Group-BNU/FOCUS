#ifndef CONTRACT_QT2_QT2_SU2_H
#define CONTRACT_QT2_QT2_SU2_H

#include "stensor2su2.h"

namespace ctns{

   // formula: qt2(r,c) = \sum_x qt2a(r,x)*qt2b(x,c) [GEMM]
   template <typename Tm>
      stensor2su2<Tm> contract_qt2_qt2(const stensor2su2<Tm>& qt2a, 
            const stensor2su2<Tm>& qt2b){
        
         // only for symmetry is zero 
         assert(qt2a.info.sym.is_zero());
         assert(qt2b.info.sym.is_zero());

         assert(qt2a.dir_col() == !qt2b.dir_row());
         assert(qt2a.info.qcol == qt2b.info.qrow);
         //qsym sym = qt2a.info.sym + qt2b.info.sym;
         qsym sym(3,0,0);

         direction2 dir = {qt2a.dir_row(), qt2b.dir_col()};
         stensor2su2<Tm> qt2(sym, qt2a.info.qrow, qt2b.info.qcol, dir);
         const Tm alpha = 1.0, beta = 1.0; 
         // loop over external indices
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            size_t off2 = qt2.info.get_offset(br,bc); 
            Tm* blk2 = qt2.data() + off2-1;
            int rdim = qt2.info.qrow.get_dim(br);
            int cdim = qt2.info.qcol.get_dim(bc);
            // qt2(r,c) = qt2a(r,x)*qt2b(x,c)
            const auto& nnzbx = qt2a.info._br2bc[br];
            for(const auto& bx : nnzbx){
               size_t off2a = qt2a.info.get_offset(br,bx);
               size_t off2b = qt2b.info.get_offset(bx,bc);
               if(off2a == 0 || off2b == 0) continue;
               const Tm* blk2a = qt2a.data() + off2a-1;
               const Tm* blk2b = qt2b.data() + off2b-1;
               int xdim = qt2a.info.qcol.get_dim(bx);
               linalg::xgemm("N", "N", rdim, cdim, xdim, alpha,
                     blk2a, rdim, blk2b, xdim, beta,
                     blk2, rdim);
            } // bx
         } // i
         return qt2;
      }

} // ctns

#endif
