#ifndef STENSOR3_H
#define STENSOR3_H

#include "qtensor3.h"
#include "fermionsign.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::dump(std::ofstream& ofs) const{
         info.dump(ofs);
         ofs.write((char*)(_data), sizeof(Tm)*info._size);
      }

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor3: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            const auto blk = (*this)(br,bc,bm);
            if(level >= 1){
               std::cout << "i=" << i << " idx=" << idx << " block[" 
                  << info.qrow.get_sym(br) << "," 
                  << info.qcol.get_sym(bc) << "," 
                  << info.qmid.get_sym(bm) << "]" 
                  << " dim0,dim1,dim2=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ","
                  << blk.dim2 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(idx));
            } // level>=1
         } // i
      }

   // fix middle index (bm,im) - bm-th block, im-idx - composite index!
   // A(l,r) = B[m](l,r)
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor2<ifab,Tm> qtensor3<ifab,Tm>::fix_mid(const std::pair<int,int> mdx) const{
         int bm = mdx.first, im = mdx.second;   
         assert(im < info.qmid.get_dim(bm));
         auto symIn = std::get<2>(info.dir) ? info.sym-info.qmid.get_sym(bm) : info.sym+info.qmid.get_sym(bm);
         qtensor2<ifab,Tm> qt2(symIn, info.qrow, info.qcol, 
               {std::get<0>(info.dir), std::get<1>(info.dir)});
         for(int br=0; br<info._rows; br++){
            for(int bc=0; bc<info._cols; bc++){
               const auto blk3 = (*this)(br,bc,bm);
               if(blk3.empty()) continue;
               auto blk2 = qt2(br,bc);
               linalg::xcopy(blk2.size(), blk3.get(im).data(), blk2.data()); 
            } // bc
         } // br
         return qt2;
      }

   // deal with fermionic sign in fermionic direct product
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::mid_signed(const double fac){
         ctns::mid_signed(info, _data, fac);
      }

   // wf[lcr](-1)^{p(l)}
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::row_signed(const double fac){
         ctns::row_signed(info, _data, fac);
      }

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::cntr_signed(const std::string block){
         ctns::cntr_signed(block, info, _data);
      }

   // Generate the sign for wf[lcr]|lcr> = wf3[lcr]|lrc> 
   // with wf3[lcr] = wf[lcr]*(-1)^{p[c]*p[r]}|lrc>
   // which is later used for wf3[l,c,r] <-> wf2[lr,c] (merge_lr)
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor3<ifab,Tm>::permCR_signed(){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = (*this)(br,bc,bm); 
            if(info.qmid.get_parity(bm)*info.qcol.get_parity(bc) == 1){
               linalg::xscal(blk3.size(), -1.0, blk3.data());
            }
         }
      }

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor3<ifab,Tm> qtensor3<ifab,Tm>::K(const int nbar) const{
         const double fpo = (nbar%2==0)? 1.0 : -1.0;
         // the symmetry is flipped
         qtensor3<ifab,Tm> qt3(info.sym.flip(), info.qrow, info.qcol, info.qmid, info.dir);
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = qt3(br,bc,bm); 
            // kramers
            const auto blkk = (*this)(br,bc,bm);
            int pt_r = info.qrow.get_parity(br);
            int pt_c = info.qcol.get_parity(bc);
            int pt_m = info.qmid.get_parity(bm);
            int mdim = info.qmid.get_dim(bm);
            // qt3[c](l,r) = blk[bar{c}](bar{l},bar{r})^*
            if(pt_m == 0){
               // c[e]
               for(int im=0; im<mdim; im++){
                  auto mat = blkk.get(im).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), fpo, mat.data(), blk3.get(im).data());
               }
            }else{
               assert(mdim%2 == 0);
               int mdim2 = mdim/2;
               // c[o],c[\bar{o}]
               for(int im=0; im<mdim2; im++){
                  auto mat = blkk.get(im+mdim2).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), fpo, mat.data(), blk3.get(im).data());
               }
               for(int im=0; im<mdim2; im++){
                  auto mat = blkk.get(im).time_reversal(pt_r, pt_c);
                  linalg::xaxpy(mat.size(), -fpo, mat.data(), blk3.get(im+mdim2).data());
               }
            } // pm
         } // i
         return qt3;
      }

} // ctns

#endif
