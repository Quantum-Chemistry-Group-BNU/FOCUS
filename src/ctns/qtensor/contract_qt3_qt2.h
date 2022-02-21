#ifndef CONTRACT_QT3_QT2_H
#define CONTRACT_QT3_QT2_H

namespace ctns{

// --- contract_qt3_qt2 ---
template <typename Tm>
stensor3<Tm> contract_qt3_qt2(const std::string cpos,
		 	      const stensor3<Tm>& qt3a, 
			      const stensor2<Tm>& qt2b,
			      const bool ifdagger=false){
   stensor3<Tm> qt3;
   if(cpos == "l"){
      qt3 = contract_qt3_qt2_l(qt3a, qt2b, ifdagger);
   }else if(cpos == "r"){
      qt3 = contract_qt3_qt2_r(qt3a, qt2b, ifdagger);
   }else if(cpos == "c"){
      qt3 = contract_qt3_qt2_c(qt3a, qt2b, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt3_qt2! cpos=" 
	        << cpos << std::endl;
      exit(1);
   }
   return qt3;
}

//  r/	m 
//   *  |     = [m](r,c) = op(r,x) A[m](x,c) = <mr|o|c>
//  x\--*--c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_l(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt3a.dir_row() == !qt2b.dir_col());
   assert(qt3a.info.qrow == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt2b.dir_row(), qt3a.dir_col(), qt3a.dir_mid()};
   stensor3<Tm> qt3(sym, qt2b.info.qrow, qt3a.info.qcol, qt3a.info.qmid, dir);
   // loop over external indices
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      // loop over contracted indices
      for(int bx=0; bx<qt3a.rows(); bx++){
         const auto& blk3a = qt3a(bx,bc,bm);
	 const auto& blk2b = qt2b(br,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
	 int mdim = qt3.info.qmid.get_dim(bm);
	 for(int im=0; im<mdim; im++){
            xgemm("N","N",1.0,blk2b,blk3a.get(im),1.0,blk3.get(im));
	 } // im
      } // bx
   } // i
   return qt3;
}

//     m  \ c/r
//     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
//  r--*--/ x/c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_r(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt3a.dir_col() == !qt2b.dir_col()); // each line is associated with one dir
   assert(qt3a.info.qcol == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt3a.dir_row(), qt2b.dir_row(), qt3a.dir_mid()};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qt2b.info.qrow, qt3a.info.qmid, dir);
   // loop over external indices
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      // loop over contracted indices
      for(int bx=0; bx<qt3a.cols(); bx++){
	 const auto& blk3a = qt3a(br,bx,bm);
	 const auto& blk2b = qt2b(bc,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
         int mdim = qt3.info.qmid.get_dim(bm);
         for(int im=0; im<mdim; im++){
            xgemm("N","T",1.0,blk3a.get(im),blk2b,1.0,blk3.get(im));
         } // im
      } // bx
   } // i
   return qt3;
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
//  r--*--c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_c(const stensor3<Tm>& qt3a, 
			 	const stensor2<Tm>& qt2,
			        const bool ifdagger=false){
   const auto& qt2b = ifdagger? qt2.H() : qt2;
   assert(qt3a.dir_mid() == !qt2b.dir_col());
   assert(qt3a.info.qmid == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt3a.dir_row(), qt3a.dir_col(), qt2b.dir_row()};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qt3a.info.qcol, qt2b.info.qrow, dir);
   // loop over external indices
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      // loop over contracted indices
      for(int bx=0; bx<qt3a.mids(); bx++){
         const auto& blk3a = qt3a(br,bc,bx);
	 const auto& blk2b = qt2b(bm,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
	 int N = blk3.dim0*blk3.dim1;
	 int xdim = qt3a.info.qmid.get_dim(bx);
	 int mdim = qt3.info.qmid.get_dim(bm);
	 for(int ix=0; ix<xdim; ix++){
	    for(int im=0; im<mdim; im++){
	       linalg::xaxpy(N, blk2b(im,ix), blk3a.get(ix).data(), blk3.get(im).data());
	    } // im
	 } // ix 
      } // bx
   } // i
   return qt3;
}

} // ctns

#endif
