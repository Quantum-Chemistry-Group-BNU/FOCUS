#ifndef CONTRACT_QT3_QT2_INFO_H
#define CONTRACT_QT3_QT2_INFO_H

namespace ctns{

// --- contract_qt3_qt2 ---
template <typename Tm>
void contract_qt3_qt2_info(const std::string cpos,
		           const stensor3<Tm>& qt3a, 
		           const stensor2<Tm>& qt2,
		           stensor3<Tm>& qt3,
		           const bool ifdagger=false){
   if(cpos == "l"){
      contract_qt3_qt2_info_l(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
		              qt3.info, qt3.data(), 1.0, false, ifdagger);
   }else if(cpos == "r"){
      contract_qt3_qt2_info_r(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
		              qt3.info, qt3.data(), 1.0, false, ifdagger);
   }else if(cpos == "c"){
      contract_qt3_qt2_info_c(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
		              qt3.info, qt3.data(), 1.0, false, ifdagger);
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
void contract_qt3_qt2_info_l(const qinfo3<Tm>& qt3a_info,
	       		     Tm* qt3a_data,	
		 	     const qinfo2<Tm>& qt2_info,
			     Tm* qt2_data,
			     qinfo3<Tm>& qt3_info,
			     Tm* qt3_data,
			     const double talpha,
			     const bool accum,
			     const bool ifdagger=false){
   const char* transa = ifdagger? "C" : "N";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm;
   for(int i=0; i<qt3_info._nnzaddr.size(); i++){
      int idx = qt3_info._nnzaddr[i];
      qt3_info._addr_unpack(idx,br,bc,bm);
      auto blk3 = qt3_info(br,bc,bm,qt3_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a_info._rows; bx++){
         const auto blk3a = qt3a_info(bx,bc,bm,qt3a_data);
         const auto blk2b = ifdagger? qt2_info(bx,br,qt2_data) : qt2_info(br,bx,qt2_data);
	 if(blk3a.empty() || blk2b.empty()) continue;
	 ifzero = false; 
         const Tm beta = accum? 1.0 : 0.0; 
	 int mdim = blk3.dim2;
	 for(int im=0; im<mdim; im++){
            xgemm(transa,"N",alpha,blk2b,blk3a.get(im),beta,blk3.get(im));
	 } // im
      } // bxuu
      if(ifzero && !accum) blk3.clear();
   } // i
}

//     m  \ c/r
//     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
//  r--*--/ x/c
template <typename Tm>
void contract_qt3_qt2_info_r(const qinfo3<Tm>& qt3a_info,
	       		     Tm* qt3a_data,	
		 	     const qinfo2<Tm>& qt2_info,
			     Tm* qt2_data,
			     qinfo3<Tm>& qt3_info,
			     Tm* qt3_data,
			     const double talpha,
			     const bool accum,
			     const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm;
   for(int i=0; i<qt3_info._nnzaddr.size(); i++){
      int idx = qt3_info._nnzaddr[i];
      qt3_info._addr_unpack(idx,br,bc,bm);
      auto blk3 = qt3_info(br,bc,bm,qt3_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a_info._cols; bx++){
	 const auto blk3a = qt3a_info(br,bx,bm,qt3a_data);
	 auto blk2b = ifdagger? qt2_info(bx,bc,qt2_data) : qt2_info(bc,bx,qt2_data);
	 if(blk3a.empty() || blk2b.empty()) continue;
	 ifzero = false;
         const Tm beta = accum? 1.0 : 0.0;
	 int mdim = blk3.dim2;
         if(ifdagger) blk2b.conjugate();
	 for(int im=0; im<mdim; im++){
            xgemm("N",transb,alpha,blk3a.get(im),blk2b,beta,blk3.get(im));
         } // im
         if(ifdagger) blk2b.conjugate();
      } // bx
      if(ifzero && !accum) blk3.clear();
   } // i
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
//  r--*--c
template <typename Tm>
void contract_qt3_qt2_info_c(const qinfo3<Tm>& qt3a_info,
	       		     Tm* qt3a_data,	
		 	     const qinfo2<Tm>& qt2_info,
			     Tm* qt2_data,
			     qinfo3<Tm>& qt3_info,
			     Tm* qt3_data,
			     const double talpha,
			     const bool accum,
			     const bool ifdagger=false){
   const char* transb = ifdagger? "N" : "T";
   const Tm alpha = static_cast<Tm>(talpha);
   int br, bc, bm;
   for(int i=0; i<qt3_info._nnzaddr.size(); i++){
      int idx = qt3_info._nnzaddr[i];
      qt3_info._addr_unpack(idx,br,bc,bm);
      auto blk3 = qt3_info(br,bc,bm,qt3_data);
      bool ifzero = true;
      // loop over contracted indices
      for(int bx=0; bx<qt3a_info._mids; bx++){
	 const auto blk3a = qt3a_info(br,bc,bx,qt3a_data);
	 auto blk2b = ifdagger? qt2_info(bx,bm,qt2_data) : qt2_info(bm,bx,qt2_data);
         if(blk3a.empty() || blk2b.empty()) continue;
	 ifzero = false;
	 const Tm beta = accum? 1.0 : 0.0;
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
      if(ifzero && !accum) blk3.clear();
   } // i
}

} // ctns

#endif
