#ifndef CONTRACT_QT3_QT2_INFO_H
#define CONTRACT_QT3_QT2_INFO_H

namespace ctns{

// --- contract_qt3_qt2 ---
template <typename Tm>
void contract_qt3_qt2_info(const std::string cpos,
		           const stensor3<Tm>& qt3a, 
		           const stensor2<Tm>& qt2b,
		           stensor3<Tm>& qt3,
		           const bool ifdagger=false){
   if(cpos == "l"){
      contract_qt3_qt2_info_l(qt3a, qt2b, qt3, ifdagger);
   }else if(cpos == "r"){
      contract_qt3_qt2_info_r(qt3a, qt2b, qt3, ifdagger);
   }else if(cpos == "c"){
      contract_qt3_qt2_info_c(qt3a, qt2b, qt3, ifdagger);
   }else{
      std::cout << "error: no such case in contract_qt3_qt2_info! cpos=" 
	        << cpos << std::endl;
      exit(1);
   }
}

//  r/	m 
//   *  |     = [m](r,c) = op(r,x) A[m](x,c) = <mr|o|c>
//  x\--*--c
template <typename Tm>
void contract_qt3_qt2_info_l(const stensor3<Tm>& qt3a, 
		 	     const stensor2<Tm>& qt2,
			     stensor3<Tm>& qt3,
			     const bool ifdagger=false){
   const char* transa = ifdagger? "C" : "N";
   const Tm alpha = 1.0;
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a.rows(); bx++){
         const auto& blk3a = qt3a(bx,bc,bm);
         const auto& blk2b = ifdagger? qt2(bx,br) : qt2(br,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
         const Tm beta = ifzero? 0.0 : 1.0;
	 ifzero = false;
	 int mdim = blk3.dim2;
	 for(int im=0; im<mdim; im++){
            xgemm(transa,"N",alpha,blk2b,blk3a.get(im),beta,blk3.get(im));
	 } // im
      } // bx
      if(ifzero) blk3.clear();
   } // i
}

//     m  \ c/r
//     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
//  r--*--/ x/c
template <typename Tm>
void contract_qt3_qt2_info_r(const stensor3<Tm>& qt3a, 
		             const stensor2<Tm>& qt2,
			     stensor3<Tm>& qt3,
			     const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = 1.0;
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a.cols(); bx++){
	 const auto& blk3a = qt3a(br,bx,bm);
	 auto blk2b = ifdagger? qt2(bx,bc) : qt2(bc,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
         const Tm beta = ifzero? 0.0 : 1.0;
         ifzero = false;
	 int mdim = blk3.dim2;
         if(ifdagger) blk2b.conjugate();
	 for(int im=0; im<mdim; im++){
            xgemm("N",transb,alpha,blk3a.get(im),blk2b,beta,blk3.get(im));
         } // im
         if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero) blk3.clear();
   } // i
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
//  r--*--c
template <typename Tm>
void contract_qt3_qt2_info_c(const stensor3<Tm>& qt3a, 
			     const stensor2<Tm>& qt2,
		             stensor3<Tm>& qt3,
			     const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = 1.0;
   int br, bc, bm;
   for(int i=0; i<qt3.info._nnzaddr.size(); i++){
      int idx = qt3.info._nnzaddr[i];
      qt3.info._addr_unpack(idx,br,bc,bm);
      auto& blk3 = qt3(br,bc,bm);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a.mids(); bx++){
         const auto& blk3a = qt3a(br,bc,bx);
	 auto blk2b = ifdagger? qt2(bx,bm) : qt2(bm,bx);
	 if(blk3a.size() == 0 || blk2b.size() == 0) continue;
         const Tm beta = ifzero? 0.0 : 1.0;
	 ifzero = false;
	 int rcdim = blk3.dim0*blk3.dim1;
	 int mdim = blk3.dim2;
	 int xdim = blk3a.dim2;
	 if(ifdagger) blk2b.conjugate();
         int LDB = ifdagger? xdim : mdim;
         linalg::xgemm("N", transb, &rcdim, &mdim, &xdim, &alpha,
                       blk3a.data(), &rcdim, blk2b.data(), &LDB, &beta,
	               blk3.data(), &rcdim);
         if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero) blk3.clear();
   } // i
}

} // ctns

#endif
