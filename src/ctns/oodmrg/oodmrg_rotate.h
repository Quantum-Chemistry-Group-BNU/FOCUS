#ifndef OODMRG_ROTATE_H
#define OODMRG_ROTATE_H

namespace ctns{

   template <bool ifab, typename Tm>
      void twodot_rotate(const std::vector<Tm>& v0, 
            const std::vector<Tm>& vr, 
            qtensor4<ifab,Tm>& wf, 
            const double theta){
         std::cout << "error: not implemented for su2 case! ifab=" << ifab << std::endl;
         assert(!ifab);
         exit(1);
      }
   template <bool ifab, typename Tm, std::enable_if_t<ifab,int> = 0>
      void twodot_rotate(const std::vector<Tm>& v0, 
            const std::vector<Tm>& vr, 
            stensor4<Tm>& wf, 
            const double theta){
         int br, bc, bm, bv;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx,br,bc,bm,bv);
            auto blk4 = wf(br,bc,bm,bv);
            /*
            if((info.qmid.get_parity(bm)+info.qver.get_parity(bv))*info.qcol.get_parity(bc) == 1){
               linalg::xscal(blk4.size(), -1.0, blk4.data());
            }
            */
            // case-1: no transformation is needed
            if((br == 0 and bc == 0) or
               (br == 1 and bc == 1) or
               (br == 2 and bc == 2) or
               (br == 3 and bc == 3)){
//               size_t offset = wf.get_offset(   
            }
         }
      }

} // ctns

#endif
