#ifndef FERMIONSIGN_H
#define FERMIONSIGN_H

#include "qinfo3.h"
#include "qinfo4.h"

namespace ctns{

   // qinfo3:

   // wf[lcr](-1)^{p(c)}
   template <typename Tm>
      void mid_signed(const qinfo3<Tm>& info, Tm* data, const double fac=1.0){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = info(br,bc,bm,data);
            double fac2 = (info.qmid.get_parity(bm)==0)? fac : -fac;
            linalg::xscal(blk3.size(), fac2, blk3.data());  
         }
      }

   // wf[lcr](-1)^{p(l)}
   template <typename Tm>
      void row_signed(const qinfo3<Tm>& info, Tm* data, const double fac=1.0){
         int br, bc, bm;
         for(int i=0; i<info._nnzaddr.size(); i++){
            int idx = info._nnzaddr[i];
            info._addr_unpack(idx,br,bc,bm);
            auto blk3 = info(br,bc,bm,data); 
            double fac2 = (info.qrow.get_parity(br)==0)? fac : -fac;
            linalg::xscal(blk3.size(), fac2, blk3.data());  
         }
      }

   template <typename Tm>
      void cntr_signed(const std::string block, const qinfo3<Tm>& info, Tm* data){
         if(block == "r"){
            int br, bc, bm;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm);
               auto blk3 = info(br,bc,bm,data); 
               // (-1)^{p(l)+p(c)}wf[l,c,r]
               int pt = info.qrow.get_parity(br) 
                  + info.qmid.get_parity(bm);
               if(pt%2 == 1) linalg::xscal(blk3.size(), -1.0, blk3.data());
            } // i
         }else if(block == "c"){
            int br, bc, bm;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm);
               auto blk3 = info(br,bc,bm,data); 
               // (-1)^{p(l)}wf[l,c,r]
               int pt = info.qrow.get_parity(br);
               if(pt%2 == 1) linalg::xscal(blk3.size(), -1.0, blk3.data());
            } // i
         } // block
      }

   // qinfo4:

   // wf[lc1c2r] = wf(row,col,mid,ver)
   template <typename Tm>
      void cntr_signed(const std::string block, const qinfo4<Tm>& info, Tm* data){
         if(block == "r"){
            int br, bc, bm, bv;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm,bv);
               auto blk4 = info(br,bc,bm,bv,data);
               // (-1)^{p(l)+p(c1)+p(c2)}wf[lc1c2r]
               int pt = info.qrow.get_parity(br)
                  + info.qmid.get_parity(bm)
                  + info.qver.get_parity(bv);
               if(pt%2 == 1) linalg::xscal(blk4.size(), -1.0, blk4.data());
            } // i
         }else if(block == "c2"){
            int br, bc, bm, bv;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm,bv);
               auto blk4 = info(br,bc,bm,bv,data);
               // (-1)^{p(l)+p(c1)}wf[lc1c2r]
               int pt = info.qrow.get_parity(br)
                  + info.qmid.get_parity(bm);
               if(pt%2 == 1) linalg::xscal(blk4.size(), -1.0, blk4.data());
            } // i
         }else if(block == "c1"){
            int br, bc, bm, bv;
            for(int i=0; i<info._nnzaddr.size(); i++){
               int idx = info._nnzaddr[i];
               info._addr_unpack(idx,br,bc,bm,bv);
               auto blk4 = info(br,bc,bm,bv,data);
               // (-1)^{p(l)}wf[lc1c2r]
               int pt = info.qrow.get_parity(br);
               if(pt%2 == 1) linalg::xscal(blk4.size(), -1.0, blk4.data());
            } // i
         } // block 
      }

} // ctns

#endif
