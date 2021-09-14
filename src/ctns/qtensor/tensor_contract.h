#ifndef TENSOR_CONTRACT_H
#define TENSOR_CONTRACT_H

#include "tensor_linalg.h"

namespace ctns{

// --- tensor linear algebra : contractions ---

//template <typename Tm>
//struct stensor2;
//template <typename Tm>
//struct stensor3;

// xgemm : C[i,k] = A[i,j] B[j,k]
template <typename Tm>
stensor2<Tm> contract_qt2_qt2(const stensor2<Tm>& qt2a, 
			      const stensor2<Tm>& qt2b){
   assert(qt2a.info.dir[1] == !qt2b.info.dir[0]);
   assert(qt2a.info.qcol == qt2b.info.qrow);
   qsym sym = qt2a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt2a.info.dir[0], qt2b.info.dir[1]};
   stensor2<Tm> qt2(sym, qt2a.info.qrow, qt2b.info.qcol, dir); 
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk2 = qt2(br,bc);
	 if(blk2.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt2a.cols(); bx++){
	    const auto& blk2a = qt2a(br,bx);
	    const auto& blk2b = qt2b(bx,bc);
	    if(blk2a.size() == 0 || blk2b.size() == 0) continue;
	    xgemm("N","N",1.0,blk2a,blk2b,1.0,blk2);
	 } // bx
      } // bc
   } // br
   return qt2;
}

//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^* Bn^T [conjugation is taken on qt3a!]
//          c--*--/ qt3b
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_cr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.info.dir  == qt3b.info.dir); // bra dir fliped implicitly
   assert(qt3a.info.qmid == qt3b.info.qmid);
   assert(qt3a.info.qcol == qt3b.info.qcol);
   qsym sym = -qt3a.info.sym + qt3b.info.sym;
   stensor2<Tm> qt2(sym, qt3a.info.qrow, qt3b.info.qrow);
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk2 = qt2(br,bc);
	 if(blk2.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt3a.cols(); bx++){
	    for(int bm=0; bm<qt3a.mids(); bm++){
	       const auto& blk3a = qt3a(br,bx,bm);
	       const auto& blk3b = qt3b(bc,bx,bm);
	       if(blk3a.size() == 0 || blk3b.size() == 0) continue;
	       for(int im=0; im<qt3a.mid_dim(bm); im++){
		  xgemm("N","C",1.0,blk3a.get(im),blk3b.get(im),1.0,blk2);
	       } // im	       
	    } // bm
	 } // bx
	 blk2.conjugate();
      } // bc
   } // br
   return qt2;
}

/*
//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lc(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   assert(qt3a.qrow == qt3b.qrow);
   assert(qt3a.qmid == qt3b.qmid);
   qsym sym = -qt3a.sym + qt3b.sym;
   stensor2<Tm> qt2(sym, qt3a.qcol, qt3b.qcol); 
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
	 auto& blk = qt2(br,bc);
	 if(blk.size() == 0) continue;
	 // loop over contracted indices
         for(int bm=0; bm<qt3a.mids(); bm++){
	    for(int bx=0; bx<qt3a.rows(); bx++){
	       const auto& blka = qt3a(bm,bx,br);
	       const auto& blkb = qt3b(bm,bx,bc);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
               for(int im=0; im<qt3a.qmid.get_dim(bm); im++){
	          blk += linalg::xgemm("C","N",blka[im],blkb[im]); 
	       } // im
	    } // bx
	 } // bm
      } // bc
   } // br
   return qt2;
}

// 	      r|
//          /--*--\ qt3a
// q(r,c) = |x    |y	  = <r|c> = tr(A[r]^* B[c]^T)
//          \--*--/ qt3b
//            c|
template <typename Tm>
stensor2<Tm> contract_qt3_qt3_lr(const stensor3<Tm>& qt3a, 
				 const stensor3<Tm>& qt3b){
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   assert(qt3a.qrow == qt3b.qrow);
   assert(qt3a.qcol == qt3b.qcol);
   qsym sym = -qt3a.sym + qt3b.sym;
   stensor2<Tm> qt2(sym, qt3a.qmid, qt3b.qmid);
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk = qt2(br,bc);
	 if(blk.size() == 0) continue;
	 // loop over contracted indices
         for(int bx=0; bx<qt3a.rows(); bx++){
	    for(int by=0; by<qt3a.cols(); by++){
	       const auto& blka = qt3a(br,bx,by);
	       const auto& blkb = qt3b(bc,bx,by);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int ic=0; ic<qt2.qcol.get_dim(bc); ic++){
                  for(int ir=0; ir<qt2.qrow.get_dim(br); ir++){
	             blk(ir,ic) += linalg::xgemm("N","T",blka[ir].conj(),blkb[ic]).trace();
		  } // ir 
	       } // ic
	    } // by
	 } // bx
      } // bc
   } // br
   return qt2;
}
*/

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
//  r--*--c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_c(const stensor3<Tm>& qt3a, 
			 	const stensor2<Tm>& qt2b){
   assert(qt3a.info.dir[2] == !qt2b.info.dir[1]);
   assert(qt3a.info.qmid == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt3a.info.dir[0], qt3a.info.dir[1], qt2b.info.dir[0]};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qt3a.info.qcol, qt2b.info.qrow, dir);
   // loop over external indices
   for(int br=0; br<qt3.rows(); br++){
      for(int bc=0; bc<qt3.cols(); bc++){
         for(int bm=0; bm<qt3.mids(); bm++){
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.mids(); bx++){
	       const auto& blk3a = qt3a(br,bc,bx);
	       const auto& blk2b = qt2b(bm,bx);
	       if(blk3a.size() == 0 || blk2b.size() == 0) continue;
	       int N = blk3.dim0*blk3.dim1;
	       for(int ix=0; ix<qt3a.mid_dim(bx); ix++){
	          for(int im=0; im<qt3.mid_dim(bm); im++){
		     linalg::xaxpy(N, blk2b(im,ix), blk3a.get(ix).data(), blk3.get(im).data());
	          } // im
	       } // ix 
	    } // bx
	 } // bm
      } // bc
   } // br
   return qt3;
}

//  r/	m 
//   *  |     = [m](r,c) = op(r,x) A[m](x,c) = <mr|o|c>
//  x\--*--c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_l(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2b){
   assert(qt3a.info.dir[0] == !qt2b.info.dir[1]);
   assert(qt3a.info.qrow == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt2b.info.dir[0], qt3a.info.dir[1], qt3a.info.dir[2]};
   stensor3<Tm> qt3(sym, qt2b.info.qrow, qt3a.info.qcol, qt3a.info.qmid, dir);
   // loop over external indices
   for(int br=0; br<qt3.rows(); br++){
      for(int bc=0; bc<qt3.cols(); bc++){
         for(int bm=0; bm<qt3.mids(); bm++){
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.rows(); bx++){
	       const auto& blk3a = qt3a(bx,bc,bm);
	       const auto& blk2b = qt2b(br,bx);
	       if(blk3a.size() == 0 || blk2b.size() == 0) continue;
	       for(int im=0; im<qt3.mid_dim(bm); im++){
		  xgemm("N","N",1.0,blk2b,blk3a.get(im),1.0,blk3.get(im));
	       } // im
	    } // bx
	 } // bm
      } // bc
   } // br
   return qt3;
}

//     m  \ c/r
//     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
//  r--*--/ x/c
template <typename Tm>
stensor3<Tm> contract_qt3_qt2_r(const stensor3<Tm>& qt3a, 
				const stensor2<Tm>& qt2b){
   assert(qt3a.info.dir[1] == !qt2b.info.dir[1]); // each line is associated with one dir
   assert(qt3a.info.qcol == qt2b.info.qcol);
   qsym sym = qt3a.info.sym + qt2b.info.sym;
   std::vector<bool> dir = {qt3a.info.dir[0], qt2b.info.dir[0], qt3a.info.dir[2]};
   stensor3<Tm> qt3(sym, qt3a.info.qrow, qt2b.info.qrow, qt3a.info.qmid, dir);
   // loop over external indices
   for(int br=0; br<qt3.rows(); br++){
      for(int bc=0; bc<qt3.cols(); bc++){
         for(int bm=0; bm<qt3.mids(); bm++){
	    auto& blk3 = qt3(br,bc,bm);
	    if(blk3.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.cols(); bx++){
	       const auto& blk3a = qt3a(br,bx,bm);
	       const auto& blk2b = qt2b(bc,bx);
	       if(blk3a.size() == 0 || blk2b.size() == 0) continue;
	       for(int im=0; im<qt3.mid_dim(bm); im++){
		  xgemm("N","T",1.0,blk3a.get(im),blk2b,1.0,blk3.get(im));
	       } // im
	    } // bx
	 } // bm
      } // bc
   } // br
   return qt3;
}

} // ctns

#endif
