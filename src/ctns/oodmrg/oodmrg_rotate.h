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
            std::vector<Tm>& vr, 
            stensor4<Tm>& wf, 
            const double theta){
         assert(v0.size() == vr.size());
         double c = std::cos(theta);
         double s = std::sin(theta);
         double c2 = c*c, s2 = s*s, cs = c*s;
         // encode the unitary matrix for the subspace by a dictionary
         std::map<std::tuple<int,int,int>,double> udict = {
            {std::make_tuple(1,0,0), c2},{std::make_tuple(0,1,0), s2},{std::make_tuple(2,3,0),-cs},{std::make_tuple(3,2,0), cs},
            {std::make_tuple(1,0,1), s2},{std::make_tuple(0,1,1), c2},{std::make_tuple(2,3,1), cs},{std::make_tuple(3,2,1),-cs},
            {std::make_tuple(1,0,2), cs},{std::make_tuple(0,1,2),-cs},{std::make_tuple(2,3,2), c2},{std::make_tuple(3,2,2), s2},
            {std::make_tuple(1,0,3),-cs},{std::make_tuple(0,1,3), cs},{std::make_tuple(2,3,3), s2},{std::make_tuple(3,2,3), c2}
         };
         // clear
         memset(vr.data(), 0, vr.size()*sizeof(Tm));
         int br, bc, bm, bv;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx,br,bc,bm,bv);
            auto blk4 = wf(br,bc,bm,bv);
            size_t size = blk4.size();
            size_t offset = wf.info.get_offset(br,bc,bm,bv);
            assert(offset > 0);
            offset -= 1;
            // let {0,1,2,3} refers to {|0>,|2>,|a>,|b>} as defined in init_phys.h
            // case-1: {(0,0)},{(1,1)},{(2,2)},{(3,3)} - no transformation is needed
            if((bm == 0 and bv == 0) or
               (bm == 1 and bv == 1) or
               (bm == 2 and bv == 2) or
               (bm == 3 and bv == 3)){
               linalg::xcopy(size, &v0[offset], &vr[offset]); 
            }
            // case-2: {(2,0),(0,2)},{(3,0),(0,3)},{(2,1),(1,2)},{(3,1),(1,3)}
            if((bm == 2 and bv == 0) or
               (bm == 3 and bv == 0) or
               (bm == 2 and bv == 1) or
               (bm == 3 and bv == 1)){
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size, c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, s, &v0[offset1], &vr[offset]); 
            }
            if((bm == 0 and bv == 2) or
               (bm == 0 and bv == 3) or 
               (bm == 1 and bv == 2) or
               (bm == 1 and bv == 3)){
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size,  c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, -s, &v0[offset1], &vr[offset]); 
            }
            // case-3: {(1,0),(0,1),(2,3),(3,2)}
            if((bm == 1 and bv == 0) or
               (bm == 0 and bv == 1) or
               (bm == 2 and bv == 3) or
               (bm == 3 and bv == 2)){
               size_t offset0 = wf.info.get_offset(br,bc,1,0);
               size_t offset1 = wf.info.get_offset(br,bc,0,1);
               size_t offset2 = wf.info.get_offset(br,bc,2,3);
               size_t offset3 = wf.info.get_offset(br,bc,3,2);
               assert(offset0 > 0 and offset1 > 0 and offset2 > 0 and offset3 > 0);
               offset0 -= 1;
               offset1 -= 1;
               offset2 -= 1;
               offset3 -= 1;
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,0)), &v0[offset0], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,1)), &v0[offset1], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,2)), &v0[offset2], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,3)), &v0[offset3], &vr[offset]);
            }
         } // i
      }

} // ctns

#endif
