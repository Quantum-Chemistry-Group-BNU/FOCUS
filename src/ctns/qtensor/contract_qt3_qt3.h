#ifndef CONTRACT_QT3_QT3_H
#define CONTRACT_QT3_QT3_H

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
template <typename Tm>
stensor2<Tm> contract_qt3_qt3(const std::string superblock,
		 	      const stensor3<Tm>& qt3a, 
			      const stensor3<Tm>& qt3b){
   stensor2<Tm> qt2;
   if(superblock == "lc"){
      qt2 = contract_qt3_qt3_lc(qt3a, qt3b);
   }else if(superblock == "cr"){
      qt2 = contract_qt3_qt3_cr(qt3a, qt3b);
   }else if(superblock == "lr"){
      qt2 = contract_qt3_qt3_lr(qt3a, qt3b);
   }else{
      std::cout << "error: no such case in contract_qt3_qt3! superblock=" 
	        << superblock << std::endl;
      exit(1);
   }
   return qt2;
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
   for(const auto& pr : qt3a.info._qblocks){
      const auto& key = pr.first;
      int bx = std::get<0>(key);
      int br = std::get<1>(key);
      int bm = std::get<2>(key);
      const auto blk3a = qt3a(bx,br,bm);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
         if(qt3b.ifNotExist(bx,bc,bm) || qt2.ifNotExist(br,bc)) continue;
         const auto blk3b = qt3b(bx,bc,bm);
         auto blk2 = qt2(br,bc);
         int mdim = blk3a.dim2;
         for(int im=0; im<mdim; im++){
            xgemm("C","N",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
         } // im
      } // bc
   } // br
}
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lc(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qmid == qt3b.info.qmid);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qcol, qt3b.info.qcol); 
   contract_qt3_qt3_info_lc(qt3a, qt3b, qt2);
   return qt2;
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
   for(const auto& pr : qt3a.info._qblocks){
      const auto& key = pr.first;
      int br = std::get<0>(key);
      int bx = std::get<1>(key);
      int bm = std::get<2>(key);
      const auto blk3a = qt3a(br,bx,bm);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
	 if(qt3b.ifNotExist(bc,bx,bm) || qt2.ifNotExist(br,bc)) continue;
	 const auto blk3b = qt3b(bc,bx,bm);
         auto blk2 = qt2(br,bc);
	 int mdim = blk3a.dim2;
	 for(int im=0; im<mdim; im++){
	    xgemm("N","C",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	 } // im	       
      } // bc
   } // i
   qt2.conjugate();
}
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_cr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir  == qt3b.info.dir); // bra dir fliped implicitly
   assert(qt3a.info.qmid == qt3b.info.qmid);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qrow, qt3b.info.qrow);
   contract_qt3_qt3_info_cr(qt3a, qt3b, qt2);
   return qt2;
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
   for(const auto& pr : qt3a.info._qblocks){
      const auto& key = pr.first;
      int bx = std::get<0>(key);
      int by = std::get<1>(key);
      int br = std::get<2>(key);
      const auto blk3a = qt3a(bx,by,br);
      // loop over bc
      for(int bc=0; bc<qt2.cols(); bc++){
         if(qt3b.ifNotExist(bx,by,bc) || qt2.ifNotExist(br,bc)) continue;
	 const auto blk3b = qt3b(bx,by,bc);
         auto blk2 = qt2(br,bc);
	 int rdim = blk2.dim0;
	 int cdim = blk2.dim1;
	 // qt2(r,c) = qt3a*(xy,r) qt3b(xy,c)
	 int xydim = blk3a.dim0*blk3a.dim1;
         linalg::xgemm("C", "N", &rdim, &cdim, &xydim, &alpha,
                       blk3a.data(), &xydim, blk3b.data(), &xydim, &beta,
	               blk2.data(), &rdim);
      } // bc
   } // i
}
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir == qt3b.info.dir); // bra dir fliped
   assert(qt3a.info.qrow == qt3b.info.qrow);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qmid, qt3b.info.qmid);
   contract_qt3_qt3_info_lr(qt3a, qt3b, qt2);
   return qt2;
}

} // ctns

#endif
