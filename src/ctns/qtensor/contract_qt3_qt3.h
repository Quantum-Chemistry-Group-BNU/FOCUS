#ifndef CONTRACT_QT3_QT3_H
#define CONTRACT_QT3_QT3_H

namespace ctns{

template <typename Tm>
stensor2<Tm> contract_qt3_qt3(const std::string superblock,
		 	      const stensor3<Tm>& qt3a, 
			      const stensor3<Tm>& qt3b){
   stensor2<Tm> qt2;
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   if(superblock == "lc"){
      qt2.init(sym, qt3a.info.qcol, qt3b.info.qcol); 
   }else if(superblock == "cr"){
      qt2.init(sym, qt3a.info.qrow, qt3b.info.qrow);
   }else if(superblock == "lr"){
      qt2.init(sym, qt3a.info.qmid, qt3b.info.qmid);
   }
   contract_qt3_qt3_info(superblock,qt3a.info, qt3a.data(), 
		   	 qt3b.info, qt3b.data(),
		         qt2.info, qt2.data());
   return qt2;
}

template <typename Tm>
void contract_qt3_qt3_info(const std::string superblock,
		           const qinfo3<Tm>& qt3a_info,
	       		   Tm* qt3a_data,	
			   const qinfo3<Tm>& qt3b_info,
			   Tm* qt3b_data,
			   qinfo2<Tm>& qt2_info,
			   Tm* qt2_data){
   assert(qt2_info.sym == -qt3a_info.sym + qt3b_info.sym);
   assert(qt3a_info.dir  == qt3b_info.dir); // bra dir fliped
   assert(qt3a_info.qrow == qt3b_info.qrow);
   assert(qt3a_info.qcol == qt3b_info.qcol);
   assert(qt3a_info.qmid == qt3b_info.qmid);
   if(superblock == "lc"){
      contract_qt3_qt3_info_lc(qt3a_info, qt3a_data, qt3b_info, qt3b_data,
		               qt2_info, qt2_data);
   }else if(superblock == "cr"){
      contract_qt3_qt3_info_cr(qt3a_info, qt3a_data, qt3b_info, qt3b_data,
		               qt2_info, qt2_data);
   }else if(superblock == "lr"){
      contract_qt3_qt3_info_lr(qt3a_info, qt3a_data, qt3b_info, qt3b_data,
		               qt2_info, qt2_data);
   }else{
      std::cout << "error: no such case in contract_qt3_qt3_info! superblock=" 
	        << superblock << std::endl;
      exit(1);
   }
}

// formula: qt2(r,c) = \sum_xm Conj[qt3a](x,r,m)*qt3b(x,c,m) [storage: qt3(L,R,C)]
//
//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
template <typename Tm>
void contract_qt3_qt3_info_lc(const qinfo3<Tm>& qt3a_info,
	       		      const Tm* qt3a_data,	
			      const qinfo3<Tm>& qt3b_info,
			      const Tm* qt3b_data,
			      const qinfo2<Tm>& qt2_info,
			      Tm* qt2_data){
   const Tm alpha = 1.0, beta = 1.0;
   // loop over qt3a
   int bx, br, bm;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,bx,br,bm);
      size_t off3a = qt3a_info._offset[idx];
      const Tm* blk3a = qt3a_data + off3a-1;
      int xdim = qt3a_info.qrow.get_dim(bx);
      int rdim = qt3a_info.qcol.get_dim(br);
      int mdim = qt3a_info.qmid.get_dim(bm);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
         size_t off3b = qt3b_info._offset[qt3b_info._addr(bx,bc,bm)];
	 if(off3b == 0) continue;
         size_t off2 = qt2_info._offset[qt2_info._addr(br,bc)];
         if(off2 == 0) continue;
         // qt2(r,c) = Conj[qt3a](x,r,m)*qt3b(x,c,m)
         const Tm* blk3b = qt3b_data + off3b-1;
	 Tm* blk2 = qt2_data + off2-1;
	 int cdim = qt2_info.qcol.get_dim(bc);
	 int xrdim = xdim*rdim;
	 int xcdim = xdim*cdim;
         for(int im=0; im<mdim; im++){
	    const Tm* blk3a_im = blk3a + im*xrdim;
	    const Tm* blk3b_im = blk3b + im*xcdim;
	    linalg::xgemm("C", "N", &rdim, &cdim, &xdim, &alpha,
			  blk3a_im, &xdim, blk3b_im, &xdim, &beta,
			  blk2, &rdim);
         } // im
      } // bc
   } // i
}
	 
// formula: qt2(r,c) = \sum_xm Conj[qt3a](r,x,m)*qt3b(c,x,m)
//
//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^* Bn^T [conjugation is taken on qt3a!]
//          c--*--/ qt3b
template <typename Tm>
void contract_qt3_qt3_info_cr(const qinfo3<Tm>& qt3a_info,
	       		      Tm* qt3a_data,	
			      const qinfo3<Tm>& qt3b_info,
			      Tm* qt3b_data,
			      qinfo2<Tm>& qt2_info,
			      Tm* qt2_data){
   const Tm alpha = 1.0, beta = 1.0;
   // loop over qt3a
   int br, bx, bm;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,br,bx,bm);
      size_t off3a = qt3a_info._offset[idx];
      const Tm* blk3a = qt3a_data + off3a-1;
      int rdim = qt3a_info.qrow.get_dim(br);
      int xdim = qt3a_info.qcol.get_dim(bx);
      int mdim = qt3a_info.qmid.get_dim(bm);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
         size_t off3b = qt3b_info._offset[qt3b_info._addr(bc,bx,bm)];
	 if(off3b == 0) continue;
	 size_t off2 = qt2_info._offset[qt2_info._addr(br,bc)];
	 if(off2 == 0) continue;
	 // qt2(r,c) = Conj[qt3a](r,x,m)*qt3b(c,x,m)
	 const Tm* blk3b = qt3b_data + off3b-1;
	 Tm* blk2 = qt2_data + off2-1;
	 int cdim = qt2_info.qcol.get_dim(bc);
         int xmdim = xdim*mdim;
	 linalg::xgemm("N", "C", &rdim, &cdim, &xmdim, &alpha,
		       blk3a, &rdim, blk3b, &cdim, &beta,
		       blk2, &rdim);
      } // bc
   } // i
   linalg::xconj(qt2_info._size, qt2_data);
}

// formula: qt2(r,c) = \sum_xy Conj[qt3a](x,y,r)*qt3b(x,y,c)
//
// 	      r|
//          /--*--\ qt3a
// q(r,c) = |x    |y	  = <r|c> = tr(A[r]^* B[c]^T)
//          \--*--/ qt3b
//            c|
template <typename Tm>
void contract_qt3_qt3_info_lr(const qinfo3<Tm>& qt3a_info,
	       		      Tm* qt3a_data,	
			      const qinfo3<Tm>& qt3b_info,
			      Tm* qt3b_data,
			      qinfo2<Tm>& qt2_info,
			      Tm* qt2_data){
   const Tm alpha = 1.0, beta = 1.0;
   // loop over qt3a
   int bx, by, br;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,bx,by,br);
      size_t off3a = qt3a_info._offset[idx];
      const Tm* blk3a = qt3a_data + off3a-1;
      int xdim = qt3a_info.qrow.get_dim(bx);
      int ydim = qt3a_info.qcol.get_dim(by);
      int rdim = qt3a_info.qmid.get_dim(br);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
         size_t off3b = qt3b_info._offset[qt3b_info._addr(bx,by,bc)];
	 if(off3b == 0) continue;
	 size_t off2 = qt2_info._offset[qt2_info._addr(br,bc)];
	 if(off2 == 0) continue;
	 // qt2(r,c) = Conj[qt3a](x,y,r)*qt3b(x,y,c)
	 const Tm* blk3b = qt3b_data + off3b-1;
	 Tm* blk2 = qt2_data + off2-1;
         int cdim = qt2_info.qcol.get_dim(bc);
	 int xydim = xdim*ydim;
         linalg::xgemm("C", "N", &rdim, &cdim, &xydim, &alpha,
                       blk3a, &xydim, blk3b, &xydim, &beta,
	               blk2, &rdim);
      } // bc
   } // i
}

} // ctns

#endif
