#ifndef STENSOR4_H
#define STENSOR4_H

#include "qtensor4.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor4<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor4: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc, bm, bv;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm,bv);
            const auto blk = (*this)(br,bc,bm,bv);
            if(level >= 1){
               std::cout << "i=" << i << " idx=" << idx << " block[" 
                  << info.qrow.get_sym(br) << "," 
                  << info.qcol.get_sym(bc) << ","
                  << info.qmid.get_sym(bm) << "," 
                  << info.qver.get_sym(bv) << "]" 
                  << " dim0,dim1,dim2,dim3=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << "," 
                  << blk.dim2 << "," 
                  << blk.dim3 << ")"
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(idx));
            } // level>=1
         } // i
      }

   // wf[lc1c2r] = wf(row,col,mid,ver)
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor4<ifab,Tm>::cntr_signed(const std::string block){
         ctns::cntr_signed(block, info, _data);
      }

   // wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]}
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      void qtensor4<ifab,Tm>::permCR_signed(){
         int br, bc, bm, bv;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm,bv);
            auto blk4 = (*this)(br,bc,bm,bv);
            if((info.qmid.get_parity(bm)+info.qver.get_parity(bv))*info.qcol.get_parity(bc) == 1){
               linalg::xscal(blk4.size(), -1.0, blk4.data());
            }
         }
      }

   // ZL20210510: application of time-reversal operation
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<y,int>>
      qtensor4<ifab,Tm> qtensor4<ifab,Tm>::K(const int nbar) const{
         const double fpo = (nbar%2==0)? 1.0 : -1.0;
         qtensor4<ifab,Tm> qt4(info.sym.flip(), info.qrow, info.qcol, info.qmid, info.qver);
         int br, bc, bm, bv;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm,bv);
            auto blk4 = qt4(br,bc,bm,bv); 
            // kramers 
            const auto blkk = (*this)(br,bc,bm,bv);
            int pt_r = info.qrow.get_parity(br);
            int pt_c = info.qcol.get_parity(bc);
            int pt_m = info.qmid.get_parity(bm);
            int pt_v = info.qver.get_parity(bv);
            int mdim = info.qmid.get_dim(bm);
            int vdim = info.qver.get_dim(bv);
            // qt4_new(c1c2)[l,r] = qt4(c1c2_bar)[l_bar,r_bar]^*
            if(pt_m == 0 && pt_v == 0){
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     auto mat = blkk.get(im,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), fpo, mat.data(), blk4.get(im,iv).data());
                  }
               }
            }else if(pt_m == 0 && pt_v == 1){
               assert(vdim%2 == 0);
               int vdim2 = vdim/2;
               for(int iv=0; iv<vdim2; iv++){
                  for(int im=0; im<mdim; im++){
                     auto mat = blkk.get(im,iv+vdim2).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), fpo, mat.data(), blk4.get(im,iv).data());
                  }
               }
               for(int iv=0; iv<vdim2; iv++){
                  for(int im=0; im<mdim; im++){
                     auto mat = blkk.get(im,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), -fpo, mat.data(), blk4.get(im,iv+vdim2).data());
                  }
               }
            }else if(pt_m == 1 && pt_v == 0){
               assert(mdim%2 == 0);
               int mdim2 = mdim/2;
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im+mdim2,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), fpo, mat.data(), blk4.get(im,iv).data());
                  }
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), -fpo, mat.data(), blk4.get(im+mdim2,iv).data());
                  }
               }
            }else if(pt_m == 1 && pt_v == 1){
               assert(mdim%2 == 0 && vdim%2 == 0);
               int mdim2 = mdim/2;
               int vdim2 = vdim/2;
               for(int iv=0; iv<vdim2; iv++){
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im+mdim2,iv+vdim2).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), fpo, mat.data(), blk4.get(im,iv).data());
                  }
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im,iv+vdim2).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), -fpo, mat.data(), blk4.get(im+mdim2,iv).data());
                  }
               }
               for(int iv=0; iv<vdim2; iv++){
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im+mdim2,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), -fpo, mat.data(), blk4.get(im,iv+vdim2).data());
                  }
                  for(int im=0; im<mdim2; im++){
                     auto mat = blkk.get(im,iv).time_reversal(pt_r, pt_c);
                     linalg::xaxpy(mat.size(), fpo, mat.data(), blk4.get(im+mdim2,iv+vdim2).data());
                  }
               }
            } // (pm,pv)
         } // i
         return qt4;
      }

} // ctns

#endif
