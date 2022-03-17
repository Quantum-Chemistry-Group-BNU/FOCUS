#ifndef CONTRACT_QT4_QT2_H
#define CONTRACT_QT4_QT2_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2,
			      const bool iftrans=false){
   stensor4<Tm> qt4;
   auto sym2 = iftrans? -qt2.info.sym : qt2.info.sym;
   qsym sym = qt4a.info.sym + sym2;
   auto qext = iftrans? qt2.info.qcol : qt2.info.qrow; 
   auto qint = iftrans? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   if(cpos == "l"){
      assert(qt4a.dir_row() == !dint);
      assert(qt4a.info.qrow == qint);
      qt4.init(sym, qext, qt4a.info.qcol, qt4a.info.qmid, qt4a.info.qver);
      contract_qt4_qt2_info_l(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		              qt4.info, qt4.data(), iftrans);
   }else if(cpos == "r"){
      assert(qt4a.dir_col() == !dint);
      assert(qt4a.info.qcol == qint);
      qt4.init(sym, qt4a.info.qrow, qext, qt4a.info.qmid, qt4a.info.qver);
      contract_qt4_qt2_info_r(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		              qt4.info, qt4.data(), iftrans);
   }else if(cpos == "c1"){
      assert(qt4a.dir_mid() == !dint);
      assert(qt4a.info.qmid == qint);
      qt4.init(sym, qt4a.info.qrow, qt4a.info.qcol, qext, qt4a.info.qver);
      contract_qt4_qt2_info_c1(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		               qt4.info, qt4.data(), iftrans);
   }else if(cpos == "c2"){
      assert(qt4a.dir_ver() == !dint);
      assert(qt4a.info.qver == qint);
      qt4.init(sym, qt4a.info.qrow, qt4a.info.qcol, qt4a.info.qmid, qext);
      contract_qt4_qt2_info_c2(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		               qt4.info, qt4.data(), iftrans);
   }else{
      std::cout << "error: no such case in contract_qt4_qt2! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
   return qt4;
}

// formula: qt4(r,c,m,v) = \sum_x qt2(r,x)*qt4a(x,c,m,v) ; iftrans=false 
// 		         = \sum_x qt2(x,r)*qt4a(x,c,m,v) ; iftrans=true
//
//					  r / m v
// sigma(r,c,m,v) = op(r,x)*wf(x,c,m,v) =  *  \ /    
//                                        x \--*--c
template <typename Tm>
void contract_qt4_qt2_info_l(const qinfo4<Tm>& qt4a_info,
	       		     const Tm* qt4a_data,	
			     const qinfo2<Tm>& qt2_info,
			     const Tm* qt2_data,
			     const qinfo4<Tm>& qt4_info,
			     Tm* qt4_data,
			     const bool iftrans=false){
   const Tm alpha = 1.0, beta = 0.0;
   const char* transa = iftrans? "T" : "N";
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      size_t off4 = qt4_info._offset[idx];
      Tm* blk4 = qt4_data + off4-1;
      int rdim = qt4_info.qrow.get_dim(br);
      int cdim = qt4_info.qcol.get_dim(bc);
      int mdim = qt4_info.qmid.get_dim(bm);
      int vdim = qt4_info.qver.get_dim(bv);
      int size = rdim*cdim*mdim*vdim;
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._rows; bx++){
	 size_t off4a = qt4a_info._offset[qt4a_info._addr(bx,bc,bm,bv)];
	 if(off4a == 0) continue;
	 int jdx = iftrans? qt2_info._addr(bx,br) : qt2_info._addr(br,bx);
	 size_t off2 = qt2_info._offset[jdx];
         if(off2 == 0) continue;
	 ifzero = false; 
         // qt4(r,c,m,v) = \sum_x qt2(r,x)*qt4a(x,c,m,v) ; iftrans=false 
         // 		 = \sum_x qt2(x,r)*qt4a(x,c,m,v) ; iftrans=true
         const Tm* blk4a = qt4a_data + off4a-1;
         const Tm* blk2 = qt2_data + off2-1;
         int xdim = qt4a_info.qrow.get_dim(bx);
         int LDA = iftrans? xdim : rdim;
         int cmvdim = cdim*mdim*vdim;
	 linalg::xgemm(transa, "N", &rdim, &cmvdim, &xdim, &alpha,
		       blk2, &LDA, blk4a, &xdim, &beta,
		       blk4, &rdim);
      } // bx
      if(ifzero) memset(blk4, 0, size*sizeof(Tm));
   } // i 
}

// formula: qt4(r,c,m,v) = \sum_x qt2(c,x)*qt4a(r,x,m,v) ; iftrans=false 
// 		         = \sum_x qt2(x,c)*qt4a(r,x,m,v) ; iftrans=true
//
//					    m v \ c
// sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v) =   \ /  * 
//                                        r--*--/ x
template <typename Tm>
void contract_qt4_qt2_info_r(const qinfo4<Tm>& qt4a_info,
	       		     const Tm* qt4a_data,	
			     const qinfo2<Tm>& qt2_info,
			     const Tm* qt2_data,
			     const qinfo4<Tm>& qt4_info,
			     Tm* qt4_data,
			     const bool iftrans=false){
   const Tm alpha = 1.0, beta = 0.0;
   const char* transb = iftrans? "N" : "T";
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      size_t off4 = qt4_info._offset[idx];
      Tm* blk4 = qt4_data + off4-1;
      int rdim = qt4_info.qrow.get_dim(br); 
      int cdim = qt4_info.qcol.get_dim(bc); 
      int mdim = qt4_info.qmid.get_dim(bm);
      int vdim = qt4_info.qver.get_dim(bv);
      int size = rdim*cdim*mdim*vdim;
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._cols; bx++){
	 size_t off4a = qt4a_info._offset[qt4a_info._addr(br,bx,bm,bv)];
         if(off4a == 0) continue;
	 int jdx = iftrans? qt2_info._addr(bx,bc) : qt2_info._addr(bc,bx);
         size_t off2 = qt2_info._offset[jdx];
	 if(off2 == 0) continue;
         ifzero = false;
         // qt4(r,c,m,v) = \sum_x qt2(c,x)*qt4a(r,x,m,v) ; iftrans=false 
         // 		 = \sum_x qt2(x,c)*qt4a(r,x,m,v) ; iftrans=true
	 const Tm* blk4a = qt4a_data + off4a-1;
	 const Tm* blk2 = qt2_data + off2-1;
	 int xdim = qt4a_info.qcol.get_dim(bx);
         int LDB = iftrans? xdim : cdim;
	 int rxdim = rdim*xdim;
	 int rcdim = rdim*cdim;
	 for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
	       const Tm* blk4a_imv = blk4a + (iv*mdim+im)*rxdim;
	       Tm* blk4_imv = blk4 + (iv*mdim+im)*rcdim;
	       linalg::xgemm("N", transb, &rdim, &cdim, &xdim, &alpha,
			     blk4a_imv, &rdim, blk2, &LDB, &beta,
			     blk4_imv, &rdim);  
            } // im
         } // iv
      } // bx
      if(ifzero) memset(blk4, 0, size*sizeof(Tm));
   } // i
}

// formula: qt4(r,c,m,v) = \sum_x qt2(m,x)*qt4a(r,c,x,v) ; iftrans=false 
// 		         = \sum_x qt2(x,m)*qt4a(r,c,x,v) ; iftrans=true
//
//					  m /
//                                         *   v 
// sigma(r,c,m,v) = op(m,x)*wf(r,c,x,v) = x \ /  
//                                        r--*--c
template <typename Tm>
void contract_qt4_qt2_info_c1(const qinfo4<Tm>& qt4a_info,
	       		      const Tm* qt4a_data,	
			      const qinfo2<Tm>& qt2_info,
			      const Tm* qt2_data,
			      const qinfo4<Tm>& qt4_info,
			      Tm* qt4_data,
			      const bool iftrans=false){
   const Tm alpha = 1.0, beta = 0.0;
   const char* transb = iftrans? "N" : "T";
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      size_t off4 = qt4_info._offset[idx];
      Tm* blk4 = qt4_data + off4-1;
      int rdim = qt4_info.qrow.get_dim(br);
      int cdim = qt4_info.qcol.get_dim(bc);
      int mdim = qt4_info.qmid.get_dim(bm);
      int vdim = qt4_info.qver.get_dim(bv);
      int size = rdim*cdim*mdim*vdim;
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._mids; bx++){
	 size_t off4a = qt4a_info._offset[qt4a_info._addr(br,bc,bx,bv)];
	 if(off4a == 0) continue;
	 int jdx = iftrans? qt2_info._addr(bx,bm) : qt2_info._addr(bm,bx);
	 size_t off2 = qt2_info._offset[jdx];
         if(off2 == 0) continue;
         ifzero = false;
         // qt4(r,c,m,v) = \sum_x qt2(m,x)*qt4a(r,c,x,v) ; iftrans=false 
         // 		 = \sum_x qt2(x,m)*qt4a(r,c,x,v) ; iftrans=true
         const Tm* blk4a = qt4a_data + off4a-1;
         const Tm* blk2 = qt2_data + off2-1;
         int xdim = qt4a_info.qmid.get_dim(bx);
	 int LDB = iftrans? xdim : mdim;
	 int rcdim = rdim*cdim;
	 int rcmdim = rcdim*mdim;
	 int rcxdim = rcdim*xdim;
         for(int iv=0; iv<vdim; iv++){
	    const Tm* blk4a_iv = blk4a + iv*rcxdim;
	    Tm* blk4_iv =  blk4 + iv*rcmdim;
            linalg::xgemm("N", transb, &rcdim, &mdim, &xdim, &alpha,
                          blk4a_iv, &rcdim, blk2, &LDB, &beta,
 	                  blk4_iv, &rcdim);
         } // iv
      } // bx
      if(ifzero) memset(blk4, 0, size*sizeof(Tm));
   } // i
}

// formula: qt4(r,c,m,v) = \sum_x qt2(v,x)*qt4a(r,c,m,x) ; iftrans=false 
// 		         = \sum_x qt2(x,v)*qt4a(r,c,m,x) ; iftrans=true
//
//					      \ v
//                                         m   *  
// sigma(r,c,m,v) = op(v,x)*wf(r,c,m,x) =   \ / x
//                                        r--*--c
template <typename Tm>
void contract_qt4_qt2_info_c2(const qinfo4<Tm>& qt4a_info,
	       		      const Tm* qt4a_data,	
			      const qinfo2<Tm>& qt2_info,
			      const Tm* qt2_data,
			      const qinfo4<Tm>& qt4_info,
			      Tm* qt4_data,
			      const bool iftrans=false){
   const Tm alpha = 1.0, beta = 0.0;
   const char* transb = iftrans? "N" : "T";
   int br, bc, bm, bv;
   for(int i=0; i<qt4_info._nnzaddr.size(); i++){
      int idx = qt4_info._nnzaddr[i];
      qt4_info._addr_unpack(idx,br,bc,bm,bv);
      size_t off4 = qt4_info._offset[idx];
      Tm* blk4 = qt4_data + off4-1;
      int rdim = qt4_info.qrow.get_dim(br);
      int cdim = qt4_info.qcol.get_dim(bc);
      int mdim = qt4_info.qmid.get_dim(bm);
      int vdim = qt4_info.qver.get_dim(bv);
      int size = rdim*cdim*mdim*vdim;
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt4a_info._vers; bx++){
	 size_t off4a = qt4a_info._offset[qt4a_info._addr(br,bc,bm,bx)];
	 if(off4a == 0) continue;
	 int jdx = iftrans? qt2_info._addr(bx,bv) : qt2_info._addr(bv,bx);
	 size_t off2 = qt2_info._offset[jdx];
         if(off2 == 0) continue;
         ifzero = false;
	 // qt4(r,c,m,v) = \sum_x qt2(v,x)*qt4a(r,c,m,x) ; iftrans=false 
	 // 		 = \sum_x qt2(x,v)*qt4a(r,c,m,x) ; iftrans=true
         const Tm* blk4a = qt4a_data + off4a-1;
         const Tm* blk2 = qt2_data + off2-1;
         int xdim = qt4a_info.qver.get_dim(bx);
	 int LDB = iftrans? xdim : vdim;
	 int rcmdim = rdim*cdim*mdim;
	 linalg::xgemm("N", transb, &rcmdim, &vdim, &xdim, &alpha,
		       blk4a, &rcmdim, blk2, &LDB, &beta,
		       blk4, &rcmdim);
      } // bx
      if(ifzero) memset(blk4, 0, size*sizeof(Tm));
   } // i
}

} // ctns

#endif
