#ifndef CONTRACT_QT4_QT2_H
#define CONTRACT_QT4_QT2_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2b,
			      const bool ifdagger=false){
   stensor4<Tm> qt4;
   if(cpos == "l"){
      qt4 = contract_qt4_qt2_l(qt4a, qt2b, ifdagger);
   }else if(cpos == "r"){
      qt4 = contract_qt4_qt2_r(qt4a, qt2b, ifdagger);
   }else if(cpos == "c1"){
      qt4 = contract_qt4_qt2_c1(qt4a, qt2b, ifdagger);
   }else if(cpos == "c2"){
      qt4 = contract_qt4_qt2_c2(qt4a, qt2b, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt4_qt2! cpos=" 
                << cpos << std::endl;
      exit(1);
   }
   return qt4;
}

//					  r / m v
// sigma(r,c,m,v) = op(r,x)*wf(x,c,m,v) =  *  \ /    
//                                        x \--*--c
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_l(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
/*
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt4a.dir_row() == !qt2b.dir_col());
   assert(qt4a.info.qrow == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt2b.info.qrow, qt4a.info.qcol, 
		    qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.rows(); bx++){
         const auto& blk4a = qt4a(bx,bc,bm,bv);
         const auto& blk2b = qt2b(br,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
         int mdim = qt4.info.qmid.get_dim(bm);
         int vdim = qt4.info.qver.get_dim(bv);
         for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               xgemm("N","N",1.0,blk2b,blk4a.get(im,iv),1.0,blk4.get(im,iv));
            } // im
         } // iv
      } // bx
   } // i
*/
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_row() == !dint);
   assert(qt4a.info.qrow == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qext, qt4a.info.qcol, qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   const char* transa = ifdagger? "C" : "N";
   const Tm alpha = 1.0, beta = 1.0;
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.rows(); bx++){
         const auto& blk4a = qt4a(bx,bc,bm,bv);
         const auto& blk2b = ifdagger? qt2(bx,br) : qt2(br,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
         int mdim = blk4.dim2;
         int vdim = blk4.dim3;
         for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               xgemm(transa,"N",1.0,blk2b,blk4a.get(im,iv),1.0,blk4.get(im,iv));
            } // im
         } // iv
      } // bx
   } // i 
   return qt4;
}

//					    m v \ c
// sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v) =   \ /  * 
//                                        r--*--/ x
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_r(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2,
				const bool ifdagger=false){
/*
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt4a.dir_col() == !qt2b.dir_col()); // each line is associated with one dir
   assert(qt4a.info.qcol == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt2b.info.qrow, 
		    qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.cols(); bx++){
         const auto& blk4a = qt4a(br,bx,bm,bv);
         const auto& blk2b = qt2b(bc,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
         // sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v)
         int mdim = qt4.info.qmid.get_dim(bm);
         int vdim = qt4.info.qver.get_dim(bv);
         for(int iv=0; iv<vdim; iv++){
            for(int im=0; im<mdim; im++){
               xgemm("N","T",1.0,blk4a.get(im,iv),blk2b,1.0,blk4.get(im,iv));
            } // im
         } // iv
      } // bx
   } // i
*/
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_col() == !dint);
   assert(qt4a.info.qcol == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qext, qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = 1.0, beta = 1.0;
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.cols(); bx++){
         const auto& blk4a = qt4a(br,bx,bm,bv);
         auto blk2b = ifdagger? qt2(bx,bc) : qt2(bc,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
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
   } // i
   return qt4;
}

//					  m /
//                                         *   v 
// sigma(r,c,m,v) = op(m,x)*wf(r,c,x,v) = x \ /  
//                                        r--*--c
//
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c1(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
/*
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt4a.dir_mid() == !qt2b.dir_col());
   assert(qt4a.info.qmid == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, 
		    qt2b.info.qrow, qt4a.info.qver);
   // loop over external indices
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.mids(); bx++){
         const auto& blk4a = qt4a(br,bc,bx,bv);
         const auto& blk2b = qt2b(bm,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
         // sigma(rc,m,v) = op(m,x)*wf(rc,x,v)
         int N = blk4.dim0*blk4.dim1;
         int xdim = qt4a.info.qmid.get_dim(bx);
         int mdim = qt4.info.qmid.get_dim(bm);
         int vdim = qt4.info.qver.get_dim(bv);
         for(int iv=0; iv<vdim; iv++){
            for(int ix=0; ix<xdim; ix++){
               for(int im=0; im<mdim; im++){
                  linalg::xaxpy(N, blk2b(im,ix), blk4a.get(ix,iv).data(), 
			        blk4.get(im,iv).data());
               } // im
            } // ix
         } // iv
      } // bx
   } // i
*/
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_mid() == !dint);
   assert(qt4a.info.qmid == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qext, qt4a.info.qver);
   // loop over external indices
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = 1.0, beta = 1.0;
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.mids(); bx++){
         const auto& blk4a = qt4a(br,bc,bx,bv);
         auto blk2b = ifdagger? qt2(bx,bm) : qt2(bm,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
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
   } // i
   return qt4;
}

//					      \ v
//                                         m   *  
// sigma(r,c,m,v) = op(v,x)*wf(r,c,m,x) =   \ / x
//                                        r--*--c
//  
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c2(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2,
				 const bool ifdagger=false){
/*
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt4a.dir_ver() == !qt2b.dir_col());
   assert(qt4a.info.qver == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, 
		    qt4a.info.qmid, qt2b.info.qrow);
   // loop over external indices
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.vers(); bx++){
         const auto& blk4a = qt4a(br,bc,bm,bx);
         const auto& blk2b = qt2b(bv,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
         // sigma(rcm,v) = op(v,x)*wf(rcm,x)
         int N = blk4.dim0*blk4.dim1*blk4.dim2;
         int xdim = qt4a.info.qver.get_dim(bx);
         int vdim = qt4.info.qver.get_dim(bv);
         for(int iv=0; iv<vdim; iv++){
            for(int ix=0; ix<xdim; ix++){
               linalg::xaxpy(N, blk2b(iv,ix), blk4a.get(ix).data(), 
			     blk4.get(iv).data());
            } // ix
         } // iv
      } // bx
   } // i
*/
   auto sym2 = ifdagger? -qt2.info.sym : qt2.info.sym;
   auto qext = ifdagger? qt2.info.qcol : qt2.info.qrow; 
   auto qint = ifdagger? qt2.info.qrow : qt2.info.qcol;
   auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
   auto dint = qt2.dir_col();
   assert(qt4a.dir_ver() == !dint);
   assert(qt4a.info.qver == qint);
   qsym sym = qt4a.info.sym + sym2;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, qt4a.info.qmid, qext);
   // loop over external indices
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = 1.0, beta = 1.0;
   int br, bc, bm, bv;
   for(int i=0; i<qt4.info._nnzaddr.size(); i++){
      int idx = qt4.info._nnzaddr[i];
      qt4.info._addr_unpack(idx,br,bc,bm,bv);
      auto& blk4 = qt4(br,bc,bm,bv);
      // loop over contracted indices
      for(int bx=0; bx<qt4a.vers(); bx++){
         const auto& blk4a = qt4a(br,bc,bm,bx);
         auto blk2b = ifdagger? qt2(bx,bv) : qt2(bv,bx);
         if(blk4a.size() == 0 || blk2b.size() == 0) continue;
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
   } // i
   return qt4;
}

} // ctns

#endif
