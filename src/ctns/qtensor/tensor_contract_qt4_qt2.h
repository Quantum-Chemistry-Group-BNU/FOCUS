#ifndef TENSOR_CONTRACT_QT4_QT2_H
#define TENSOR_CONTRACT_QT4_QT2_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2b,
			      const bool ifdagger=false){
   const auto& qt2 = ifdagger? qt2b.H() : qt2b;
   stensor4<Tm> qt4 = qt4a;
   if(cpos == "l"){
      qt4 = contract_qt4_qt2_l(qt4a, qt2);
   }else if(cpos == "r"){
      qt4 = contract_qt4_qt2_r(qt4a, qt2);
   }else if(cpos == "c1"){
      qt4 = contract_qt4_qt2_c1(qt4a, qt2);
   }else if(cpos == "c2"){
      qt4 = contract_qt4_qt2_c2(qt4a, qt2);
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
				const stensor2<Tm>& qt2b){
   assert(qt4a.dir_row() == !qt2b.dir_col());
   assert(qt4a.info.qrow == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt2b.info.qrow, qt4a.info.qcol, 
		    qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   for(int br=0; br<qt4.rows(); br++){
      for(int bc=0; bc<qt4.cols(); bc++){
         for(int bm=0; bm<qt4.mids(); bm++){
 	    for(int bv=0; bv<qt4.vers(); bv++){
  	       auto& blk4 = qt4(br,bc,bm,bv);
  	       if(blk4.size() == 0) continue;
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
	    } // bv
	 } // bm
      } // bc
   } // br
   return qt4;
}

//					    m v \ c
// sigma(r,c,m,v) = op(c,x)*wf(r,x,m,v) =   \ /  * 
//                                        r--*--/ x
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_r(const stensor4<Tm>& qt4a, 
				const stensor2<Tm>& qt2b){
   assert(qt4a.dir_col() == !qt2b.dir_col()); // each line is associated with one dir
   assert(qt4a.info.qcol == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt2b.info.qrow, 
		    qt4a.info.qmid, qt4a.info.qver);
   // loop over external indices
   for(int br=0; br<qt4.rows(); br++){
      for(int bc=0; bc<qt4.cols(); bc++){
         for(int bm=0; bm<qt4.mids(); bm++){
 	    for(int bv=0; bv<qt4.vers(); bv++){
  	       auto& blk4 = qt4(br,bc,bm,bv);
  	       if(blk4.size() == 0) continue;
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
  	    } // bv
	 } // bm
      } // bc
   } // br
   return qt4;
}

//					  m /
//                                         *   v 
// sigma(r,c,m,v) = op(m,x)*wf(r,c,x,v) = x \ /  
//                                        r--*--c
//
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c1(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2b){
   assert(qt4a.dir_mid() == !qt2b.dir_col());
   assert(qt4a.info.qmid == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, 
		    qt2b.info.qrow, qt4a.info.qver);
   // loop over external indices
   for(int br=0; br<qt4.rows(); br++){
      for(int bc=0; bc<qt4.cols(); bc++){
         for(int bm=0; bm<qt4.mids(); bm++){
	    for(int bv=0; bv<qt4.vers(); bv++){
	       auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
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
	                   linalg::xaxpy(N, blk2b(im,ix), blk4a.get(ix,iv).data(), blk4.get(im,iv).data());
	                } // im
		     } // ix
	          } // iv
	       } // bx
	    } // bv
	 } // bm
      } // bc
   } // br
   return qt4;
}

//					      \ v
//                                         m   *  
// sigma(r,c,m,v) = op(v,x)*wf(r,c,m,x) =   \ / x
//                                        r--*--c
//  
template <typename Tm>
stensor4<Tm> contract_qt4_qt2_c2(const stensor4<Tm>& qt4a, 
			 	 const stensor2<Tm>& qt2b){
   assert(qt4a.dir_ver() == !qt2b.dir_col());
   assert(qt4a.info.qver == qt2b.info.qcol);
   qsym sym = qt4a.info.sym + qt2b.info.sym;
   stensor4<Tm> qt4(sym, qt4a.info.qrow, qt4a.info.qcol, 
		    qt4a.info.qmid, qt2b.info.qrow);
   // loop over external indices
   for(int br=0; br<qt4.rows(); br++){
      for(int bc=0; bc<qt4.cols(); bc++){
         for(int bm=0; bm<qt4.mids(); bm++){
	    for(int bv=0; bv<qt4.vers(); bv++){
	       auto& blk4 = qt4(br,bc,bm,bv);
	       if(blk4.size() == 0) continue;
	       // loop over contracted indices
	       for(int bx=0; bx<qt4a.mids(); bx++){
	          const auto& blk4a = qt4a(br,bc,bm,bx);
	          const auto& blk2b = qt2b(bv,bx);
	          if(blk4a.size() == 0 || blk2b.size() == 0) continue;
		  // sigma(rcm,v) = op(v,x)*wf(rcm,x)
	          int N = blk4.dim0*blk4.dim1*blk4.dim2;
	          int xdim = qt4a.info.qver.get_dim(bx);
  	          int vdim = qt4.info.qver.get_dim(bv);
  	          for(int iv=0; iv<vdim; iv++){
	             for(int ix=0; ix<xdim; ix++){
	                linalg::xaxpy(N, blk2b(iv,ix), blk4a.get(ix).data(), blk4.get(iv).data());
		     } // ix
	          } // iv
	       } // bx
	    } // bv
	 } // bm
      } // bc
   } // br
   return qt4;
}

} // ctns

#endif
