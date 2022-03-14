#ifndef CONTRACT_QT4_QT2_H
#define CONTRACT_QT4_QT2_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2,
			      const bool ifdagger=false){
   stensor4<Tm> qt4;
   if(cpos == "l"){
      qt4 = contract_qt4_qt2_l(qt4a, qt2, ifdagger);
   }else if(cpos == "r"){
      qt4 = contract_qt4_qt2_r(qt4a, qt2, ifdagger);
   }else if(cpos == "c1"){
      qt4 = contract_qt4_qt2_c1(qt4a, qt2, ifdagger);
   }else if(cpos == "c2"){
      qt4 = contract_qt4_qt2_c2(qt4a, qt2, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt4_qt2! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_l(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_row() == !dint);
   assert(qt4a.info.qrow == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qext, qt4a.info.qcol, qt4a.info.qmid, qt4a.info.qver);
   contract_qt4_qt2_info_l(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		           qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_r(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_col() == !dint);
   assert(qt4a.info.qcol == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qext, qt4a.info.qmid, qt4a.info.qver);
   contract_qt4_qt2_info_r(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		           qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c1(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_mid() == !dint);
   assert(qt4a.info.qmid == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qext, qt4a.info.qver);
   contract_qt4_qt2_info_c1(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		            qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c2(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_ver() == !dint);
   assert(qt4a.info.qver == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qt4a.info.qmid, qext);
   contract_qt4_qt2_info_c2(qt4a.info, qt4a.data(), qt2.info, qt2.data(), 
		            qt4.info, qt4.data(), 1.0, false, ifdagger);
   return qt4;
}

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
/*
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
*/
   Tm* qt2b_data;
   if(ifdagger && tools::is_complex<Tm>()){
      int N = qt2_info._size;
      qt2b_data = new Tm[N];
      linalg::xcopy(N, qt2_data, qt2b_data);
      linalg::zlacgv(N, qt2b_data); 
   }else{
      qt2b_data = qt2_data;
   }   
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   const Tm beta = accum? 1.0 : 0.0;
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
	 int jdx = ifdagger? qt2_info._addr(bx,bc) : qt2_info._addr(bc,bx);
         size_t off2b = qt2_info._offset[jdx];
	 if(off2b == 0) continue;
         ifzero = false;
         // sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v) or Conj[op](x,c)*wf(r,x,m,v)
	 Tm* blk4a = qt4a_data + off4a-1;
	 Tm* blk2b = qt2_data + off2b-1;
	 int xdim = qt4a_info.qcol.get_dim(bx);
         int LDB = ifdagger? xdim : cdim;
	 int rxdim = rdim*xdim;
	 int rcdim = rdim*cdim;
	 for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               //xgemm("N",transb,alpha,blk4a.get(im,iv),blk2b,beta,blk4.get(im,iv));
	       Tm* blk4_imv = blk4 + (iv*mdim+im)*rcdim;
	       Tm* blk4a_imv = blk4a + (iv*mdim+im)*rxdim;
	       linalg::xgemm("N", transb, &rdim, &cdim, &xdim, &alpha,
			     blk4a_imv, &rdim, blk2b, &LDB, &beta,
			     blk4_imv, &rdim);  
            } // im
         } // iv
      } // bx
      if(ifzero && !accum) memset(blk4, 0, size*sizeof(Tm));
   } // i
         //if(ifdagger) blk2b.conjugate();
	 //if(ifdagger) blk2b.conjugate();
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


} // ctns

#endif
