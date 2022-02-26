#ifndef CONTRACT_QT3_QT3_INFO_H
#define CONTRACT_QT3_QT3_INFO_H

namespace ctns{

// --- contract_qt3_qt3 ---
template <typename Tm>
void contract_qt3_qt3_info(const std::string superblock,
		 	   const stensor3<Tm>& qt3a, 
			   const stensor3<Tm>& qt3b,
			   stensor2<Tm>& qt2){
   if(superblock == "lc"){
      contract_qt3_qt3_info_lc(qt3a, qt3b, qt2);
   }else if(superblock == "cr"){
      contract_qt3_qt3_info_cr(qt3a, qt3b, qt2);
   }else if(superblock == "lr"){
      contract_qt3_qt3_info_lr(qt3a, qt3b, qt2);
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
void contract_qt3_qt3_info_lc(const stensor3<Tm>& qt3a, 
		              const stensor3<Tm>& qt3b,
			      stensor2<Tm>& qt2){
   assert(qt2.info.sym == -qt3a.info.sym + qt3b.info.sym);
   qt2.clear();
   // loop over qt3a
   int bx, br, bm;
   for(int i=0; i<qt3a.info._nnzaddr.size(); i++){
      int idx = qt3a.info._nnzaddr[i];
      qt3a.info._addr_unpack(idx,bx,br,bm);
      const auto& blk3a = qt3a(bx,br,bm);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
         const auto& blk3b = qt3b(bx,bc,bm);
         auto& blk2 = qt2(br,bc);
         if(blk3b.size() == 0 || blk2.size() == 0) continue;
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
void contract_qt3_qt3_info_cr(const stensor3<Tm>& qt3a, 
			      const stensor3<Tm>& qt3b,
			      stensor2<Tm>& qt2){
   assert(qt2.info.sym == -qt3a.info.sym + qt3b.info.sym);
   qt2.clear();
   // loop over qt3a
   int br, bx, bm;
   for(int i=0; i<qt3a.info._nnzaddr.size(); i++){
      int idx = qt3a.info._nnzaddr[i];
      qt3a.info._addr_unpack(idx,br,bx,bm);
      const auto& blk3a = qt3a(br,bx,bm);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
	 const auto& blk3b = qt3b(bc,bx,bm);
         auto& blk2 = qt2(br,bc);
	 if(blk3b.size() == 0 || blk2.size() == 0) continue;
	 int mdim = blk3a.dim2;
	 for(int im=0; im<mdim; im++){
	    xgemm("N","C",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	 } // im	       
      } // bc
   } // i
   qt2.conjugate();
}

// 	      r|
//          /--*--\ qt3a
// q(r,c) = |x    |y	  = <r|c> = tr(A[r]^* B[c]^T)
//          \--*--/ qt3b
//            c|
template <typename Tm>
void contract_qt3_qt3_info_lr(const stensor3<Tm>& qt3a, 
			      const stensor3<Tm>& qt3b,
			      stensor2<Tm>& qt2){
   assert(qt2.info.sym == -qt3a.info.sym + qt3b.info.sym);
   qt2.clear();
   const Tm alpha = 1.0, beta = 1.0;
   // loop over qt3a
   int bx, by, br;
   for(int i=0; i<qt3a.info._nnzaddr.size(); i++){
      int idx = qt3a.info._nnzaddr[i];
      qt3a.info._addr_unpack(idx,bx,by,br);
      const auto& blk3a = qt3a(bx,by,br);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
	 const auto& blk3b = qt3b(bx,by,bc);
         auto& blk2 = qt2(br,bc);
         if(blk3b.size() == 0 || blk2.size() == 0) continue;
	 int rdim = blk2.dim0;
	 int cdim = blk2.dim1;
	 // qt2(r,c) = qt3a*(x,y,r) qt3b(x,y,c)
	 int xydim = blk3a.dim0*blk3a.dim1;
         linalg::xgemm("N", "C", &rdim, &cdim, &xydim, &alpha,
                       blk3a.data(), &xydim, blk3b.data(), &xydim, &beta,
	               blk2.data(), &rdim);
         blk2.conjugate();
      } // bc
   } // i
}

} // ctns

#endif
