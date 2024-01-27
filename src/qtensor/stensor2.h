#ifndef STENSOR2_H
#define STENSOR2_H

#include "qtensor2.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor2<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor2: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc);
            const auto blk = (*this)(br,bc);
            if(level >= 1){
               std::cout << "i=" << i << " idx=" << idx << " block["  
                  << br << ":" << info.qrow.get_sym(br) << "," 
                  << bc << ":" << info.qcol.get_sym(bc) << "]" 
                  << " dim0,dim1=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(idx));
            } // level>=1
         } // i
      }

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      linalg::matrix<Tm> qtensor2<ifab,Tm>::to_matrix() const{
         int m = info.qrow.get_dimAll();
         int n = info.qcol.get_dimAll();
         linalg::matrix<Tm> mat(m,n);
         // assign block to proper place
         auto roff = info.qrow.get_offset();
         auto coff = info.qcol.get_offset();
         for(int br=0; br<info._rows; br++){
            int offr = roff[br];		 
            for(int bc=0; bc<info._cols; bc++){
               int offc = coff[bc];
               const auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               for(int ic=0; ic<blk.dim1; ic++){
                  for(int ir=0; ir<blk.dim0; ir++){
                     mat(offr+ir,offc+ic) = blk(ir,ic);
                  } // ir
               } // ic
            } // bc
         } // br
         return mat;
      }

   // from dense matrix: assign block to proper place
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor2<ifab,Tm>::from_matrix(const linalg::matrix<Tm>& mat){
         auto roff = info.qrow.get_offset();
         auto coff = info.qcol.get_offset();
         for(int br=0; br<info._rows; br++){
            int offr = roff[br];		 
            for(int bc=0; bc<info._cols; bc++){
               int offc = coff[bc];
               auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               for(int ic=0; ic<blk.dim1; ic++){
                  for(int ir=0; ir<blk.dim0; ir++){
                     blk(ir,ic) = mat(offr+ir,offc+ic);
                  } // ir
               } // ic
            } // bc
         } // br
      }

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      double qtensor2<ifab,Tm>::check_identityMatrix(const double thresh_ortho, const bool debug) const{
         if(debug) std::cout << "qtensor2::check_identityMatrix thresh_ortho=" << thresh_ortho << std::endl;
         double maxdiff = -1.0;
         for(int br=0; br<info._rows; br++){
            for(int bc=0; bc<info._cols; bc++){
               const auto blk = (*this)(br,bc);
               if(blk.empty()) continue;
               if(br != bc){
                  std::string msg = "error: not a block-diagonal matrix! br,bc=";
                  tools::exit(msg+std::to_string(br)+","+std::to_string(bc));
               }
               auto qr = info.qrow.get_sym(br);
               int ndim = info.qrow.get_dim(br);
               double diff = (blk.to_matrix() - linalg::identity_matrix<Tm>(ndim)).normF();
               maxdiff = std::max(diff,maxdiff);
               if(debug || (!debug && diff > thresh_ortho)){ 
                  std::cout << " br=" << br << " qr=" << qr << " ndim=" << ndim 
                     << " |Sr-Id|_F=" << diff << std::endl;
               }
               if(diff > thresh_ortho){
                  blk.print("diagonal block");
                  tools::exit("error: not an identity matrix!"); 
               }
            } // bc
         } // br
         return maxdiff;
      }

   // ZL20200531: Permute the line of diagrams, while maintaining their directions
   // 	       This does not change the tensor, but just permute order of index
   //         
   //           i --<--*--<-- j => j -->--*-->-- i
   //
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::P() const{
         qtensor2<ifab,Tm> qt2(info.sym, info.qcol, info.qrow, 
               {std::get<1>(info.dir), std::get<0>(info.dir)});
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            int idx = qt2.info._nnzaddr[i];
            qt2.info._addr_unpack(idx,br,bc);
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

   // ZL20200531: This is used in taking Hermitian conjugate of operators.
   // 	       If row/col is permuted while dir fixed, this effectively changes 
   // 	       the direction of lines in diagrams
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::H() const{
         // symmetry of operator get changed in consistency with line changes
         qtensor2<ifab,Tm> qt2(-info.sym, info.qcol, info.qrow, info.dir);
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            int idx = qt2.info._nnzaddr[i];
            qt2.info._addr_unpack(idx,br,bc);
            auto blk = qt2(br,bc);
            // conjugate transpose
            const auto blkh = (*this)(bc,br);
            for(int ic=0; ic<blk.dim1; ic++){
               for(int ir=0; ir<blk.dim0; ir++){
                  blk(ir,ic) = tools::conjugate(blkh(ic,ir));
               } // ir
            } // ic
         } // i
         return qt2; 
      }

   // generate matrix representation for Kramers paired operators
   // suppose row and col are KRS-adapted basis, then
   //    <r|\bar{O}|c> = (K<r|\bar{O}|c>)*
   //    		    = p{O} <\bar{r}|O|\bar{c}>*
   // using \bar{\bar{O}} = p{O} O (p{O}: 'parity' of operator)
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor2<ifab,Tm> qtensor2<ifab,Tm>::K(const int nbar) const{
         const double fpo = (nbar%2==0)? 1.0 : -1.0;
         // the symmetry is flipped
         qtensor2<ifab,Tm> qt2(info.sym.flip(), info.qrow, info.qcol, info.dir);
         int br, bc;
         for(int i=0; i<qt2.info._nnzaddr.size(); i++){
            int idx = qt2.info._nnzaddr[i];
            qt2.info._addr_unpack(idx,br,bc);
            auto blk2 = qt2(br,bc);
            // kramers 
            const auto blkk = (*this)(br,bc);
            int pt_r = info.qrow.get_parity(br);
            int pt_c = info.qcol.get_parity(bc);
            auto mat = blkk.time_reversal(pt_r, pt_c);
            linalg::xaxpy(mat.size(), fpo, mat.data(), blk2.data());
         } // i
         return qt2; 
      }

} // ctns

#endif
