#ifndef CONTRACT_QT3_QT3_INFO_H
#define CONTRACT_QT3_QT3_INFO_H

namespace ctns{

// --- contract_qt3_qt3 ---
template <typename Tm>
void contract_qt3_qt3_info(const std::string superblock,
		           const qinfo3<Tm>& qt3a_info,
	       		   Tm* qt3a_data,	
			   const qinfo3<Tm>& qt3b_info,
			   Tm* qt3b_data,
			   qinfo2<Tm>& qt2_info,
			   Tm* qt2_data){
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

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
template <typename Tm>
void contract_qt3_qt3_info_lc(const qinfo3<Tm>& qt3a_info,
	       		      Tm* qt3a_data,	
			      const qinfo3<Tm>& qt3b_info,
			      Tm* qt3b_data,
			      qinfo2<Tm>& qt2_info,
			      Tm* qt2_data){
   assert(qt2_info.sym == -qt3a_info.sym + qt3b_info.sym);
   memset(qt2_data, 0, qt2_info._size*sizeof(Tm));
   // loop over qt3a
   int bx, br, bm;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,bx,br,bm);
      const auto blk3a = qt3a_info(bx,br,bm,qt3a_data);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
         const auto blk3b = qt3b_info(bx,bc,bm,qt3b_data);
         auto blk2 = qt2_info(br,bc,qt2_data);
         if(blk3b.empty() || blk2.empty()) continue;
         int mdim = blk3a.dim2;
         for(int im=0; im<mdim; im++){
            xgemm("C","N",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
         } // im
      } // bc
   } // br
}

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
   assert(qt2_info.sym == -qt3a_info.sym + qt3b_info.sym);
   memset(qt2_data, 0, qt2_info._size*sizeof(Tm));
   // loop over qt3a
   int br, bx, bm;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,br,bx,bm);
      const auto blk3a = qt3a_info(br,bx,bm,qt3a_data);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
	 const auto blk3b = qt3b_info(bc,bx,bm,qt3b_data);
         auto blk2 = qt2_info(br,bc,qt2_data);
	 if(blk3b.empty() || blk2.empty()) continue;
	 int mdim = blk3a.dim2;
	 for(int im=0; im<mdim; im++){
	    xgemm("N","C",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	 } // im	       
      } // bc
   } // i
   std::transform(qt2_data, qt2_data+qt2_info._size, qt2_data,
		  [](const Tm& x){ return tools::conjugate(x); });
}

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
   assert(qt2_info.sym == -qt3a_info.sym + qt3b_info.sym);
   memset(qt2_data, 0, qt2_info._size*sizeof(Tm));
   const Tm alpha = 1.0, beta = 1.0;
   // loop over qt3a
   int bx, by, br;
   for(int i=0; i<qt3a_info._nnzaddr.size(); i++){
      int idx = qt3a_info._nnzaddr[i];
      qt3a_info._addr_unpack(idx,bx,by,br);
      const auto blk3a = qt3a_info(bx,by,br,qt3a_data);
      // loop over bc
      for(int bc=0; bc<qt2_info._cols; bc++){
	 const auto blk3b = qt3b_info(bx,by,bc,qt3b_data);
         auto blk2 = qt2_info(br,bc,qt2_data);
         if(blk3b.empty() || blk2.empty()) continue;
	 int rdim = blk2.dim0;
	 int cdim = blk2.dim1;
	 // qt2(r,c) = \sum_xy qt3a*(xy,r) qt3b(xy,c)
	 int xydim = blk3a.dim0*blk3a.dim1;
         linalg::xgemm("C", "N", &rdim, &cdim, &xydim, &alpha,
                       blk3a.data(), &xydim, blk3b.data(), &xydim, &beta,
	               blk2.data(), &rdim);
      } // bc
   } // i
}

} // ctns

#endif
