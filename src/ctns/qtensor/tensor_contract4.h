#ifndef TENSOR_CONTRACT4_H
#define TENSOR_CONTRACT4_H

namespace ctns{

// --- contract_qt4_qt2 ---
template <typename Tm>
stensor4<Tm> contract_qt4_qt2(const std::string cpos,
		 	      const stensor4<Tm>& qt4a, 
			      const stensor2<Tm>& qt2b,
			      const bool ifdagger=false){
   const auto& qt2 = ifdagger? qt2b.H() : qt2b;
   stensor4<Tm> qt4 = qt4a;
/*
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
*/
   return qt4;
}

//  r/ m v
//   * \ /   = (r,c,m,v) = op(r,x) A(x,c,m,v)
//  x\--*--c
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

} // ctns

#endif
