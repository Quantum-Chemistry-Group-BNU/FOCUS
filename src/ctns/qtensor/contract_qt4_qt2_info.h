#ifndef CONTRACT_QT4_QT2_INFO_H
#define CONTRACT_QT4_QT2_INFO_H

namespace ctns{

//					  r / m v
// sigma(r,c,m,v) = op(r,x)*wf(x,c,m,v) =  *  \ /    
//                                        x \--*--c
template <typename Tm>
void contract_qt4_qt2_info_l(const qinfo4<Tm>& qt4a_info,
	       		     Tm* qt4a_data,	
			     const qinfo2<Tm>& qt2_info,
			     Tm* qt2_data,
			     qinfo4<Tm>& qt4_info,
			     Tm* qt4_data,
			     const double talpha,
			     const bool accum,
			     const bool ifdagger=false){
   const char* transa = ifdagger? "C" : "N";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      auto blk4 = qt4_info(br,bc,bm,bv,qt4_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._rows; bx++){
         const auto blk4a = qt4a_info(bx,bc,bm,bv,qt4a_data);
         const auto blk2b = ifdagger? qt2_info(bx,br,qt2_data) : qt2_info(br,bx,qt2_data);
         if(blk4a.empty() || blk2b.empty()) continue;
	 ifzero = false;
	 const Tm beta = accum? 1.0 : 0.0;
         int mdim = blk4.dim2;
         int vdim = blk4.dim3;
         for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               xgemm(transa,"N",alpha,blk2b,blk4a.get(im,iv),beta,blk4.get(im,iv));
            } // im
         } // iv
      } // bx
      if(ifzero && !accum) blk4.clear();
   } // i 
}

//					    m v \ c
// sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v) =   \ /  * 
//                                        r--*--/ x
template <typename Tm>
void contract_qt4_qt2_info_r(const qinfo4<Tm>& qt4a_info,
	       		     Tm* qt4a_data,	
			     const qinfo2<Tm>& qt2_info,
			     Tm* qt2_data,
			     qinfo4<Tm>& qt4_info,
			     Tm* qt4_data,
			     const double talpha,
			     const bool accum,
			     const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      auto blk4 = qt4_info(br,bc,bm,bv,qt4_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._cols; bx++){
         const auto blk4a = qt4a_info(br,bx,bm,bv,qt4a_data);
         auto blk2b = ifdagger? qt2_info(bx,bc,qt2_data) : qt2_info(bc,bx,qt2_data);
         if(blk4a.empty() || blk2b.empty()) continue;
         ifzero = false;
         const Tm beta = accum? 1.0 : 0.0;
         // sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v)
         int mdim = blk4.dim2;
         int vdim = blk4.dim3;
         if(ifdagger) blk2b.conjugate();
         for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               xgemm("N",transb,alpha,blk4a.get(im,iv),blk2b,beta,blk4.get(im,iv));
            } // im
         } // iv
	 if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero && !accum) blk4.clear();
   } // i
}

//					  m /
//                                         *   v 
// sigma(r,c,m,v) = op(m,x)*wf(r,c,x,v) = x \ /  
//                                        r--*--c
template <typename Tm>
void contract_qt4_qt2_info_c1(const qinfo4<Tm>& qt4a_info,
	       		      Tm* qt4a_data,	
			      const qinfo2<Tm>& qt2_info,
			      Tm* qt2_data,
			      qinfo4<Tm>& qt4_info,
			      Tm* qt4_data,
			      const double talpha,
			      const bool accum,
			      const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      auto blk4 = qt4_info(br,bc,bm,bv,qt4_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._mids; bx++){
         const auto blk4a = qt4a_info(br,bc,bx,bv,qt4a_data);
         auto blk2b = ifdagger? qt2_info(bx,bm,qt2_data) : qt2_info(bm,bx,qt2_data);
         if(blk4a.empty() || blk2b.empty()) continue;
         ifzero = false;
         const Tm beta = accum? 1.0 : 0.0;
         // sigma(rc,m,v) = op(m,x)*wf(rc,x,v)
         int rcdim = blk4.dim0*blk4.dim1;
         int mdim = blk4.dim2;
         int xdim = blk4a.dim2;
         int vdim = blk4.dim3;
	 if(ifdagger) blk2b.conjugate();
         int LDB = ifdagger? xdim : mdim;
         for(int iv=0; iv<vdim; iv++){
            linalg::xgemm("N", transb, &rcdim, &mdim, &xdim, &alpha,
                          blk4a.get(iv).data(), &rcdim, blk2b.data(), &LDB, &beta,
 	                  blk4.get(iv).data(), &rcdim);
         } // iv
	 if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero && !accum) blk4.clear();
   } // i
}

//					      \ v
//                                         m   *  
// sigma(r,c,m,v) = op(v,x)*wf(r,c,m,x) =   \ / x
//                                        r--*--c
template <typename Tm>
void contract_qt4_qt2_info_c2(const qinfo4<Tm>& qt4a_info,
	       		      Tm* qt4a_data,	
			      const qinfo2<Tm>& qt2_info,
			      Tm* qt2_data,
			      qinfo4<Tm>& qt4_info,
			      Tm* qt4_data,
			      const double talpha,
			      const bool accum,
			      const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      auto blk4 = qt4_info(br,bc,bm,bv,qt4_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._vers; bx++){
         const auto blk4a = qt4a_info(br,bc,bm,bx,qt4a_data);
         auto blk2b = ifdagger? qt2_info(bx,bv,qt2_data) : qt2_info(bv,bx,qt2_data);
         if(blk4a.empty() || blk2b.empty()) continue;
         ifzero = false;
         const Tm beta = accum? 1.0 : 0.0;
         // sigma(rcm,v) = op(v,x)*wf(rcm,x)
         int rcmdim = blk4.dim0*blk4.dim1*blk4.dim2;
         int xdim = blk4a.dim3;
         int vdim = blk4.dim3;
	 if(ifdagger) blk2b.conjugate();
         int LDB = ifdagger? xdim : vdim;
	 linalg::xgemm("N", transb, &rcmdim, &vdim, &xdim, &alpha,
		       blk4a.data(), &rcmdim, blk2b.data(), &LDB, &beta,
		       blk4.data(), &rcmdim);
         if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero && !accum) blk4.clear();
   } // i
}

} // ctns

#endif
