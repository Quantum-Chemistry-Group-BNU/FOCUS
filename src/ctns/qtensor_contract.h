#ifndef QTENSOR_CONTRACT_H
#define QTENSOR_CONTRACT_H

/*
#include <vector>
#include <string>
#include <map>
#include "../core/matrix.h"
#include "ctns_qsym.h"
*/

namespace ctns{

// --- tensor linear algebra : contractions ---

template <typename Tm>
struct qtensor2;
template <typename Tm>
struct qtensor3;

// xgemm : C[i,k] = A[i,j] B[j,k]
template <typename Tm>
qtensor2<Tm> contract_qt2_qt2(const qtensor2<Tm>& qt2a, 
			      const qtensor2<Tm>& qt2b){
   assert(qt2a.dir[1] == !qt2b.dir[0]);
   assert(qt2a.qcol == qt2b.qrow);
   qsym sym = qt2a.sym + qt2b.sym;
   std::vector<bool> dir = {qt2a.dir[0], qt2b.dir[1]};
   qtensor2<Tm> qt2(sym, qt2a.qrow, qt2b.qcol, dir); 
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk = qt2(br,bc);
	 if(blk.size() == 0) continue;
	 // loop over contracted indices
	 for(int bx=0; bx<qt2a.cols(); bx++){
	    const auto& blka = qt2a(br,bx);
	    const auto& blkb = qt2b(bx,bc);
	    if(blka.size() == 0 || blkb.size() == 0) continue;
	    blk += linalg::xgemm("N","N",blka,blkb);
	 } // bx
      } // bc
   } // br
   return qt2;
}

//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^* Bn^T [conjugation is taken on qt3a!]
//          c--*--/ qt3b
template <typename Tm>
qtensor2<Tm> contract_qt3_qt3_cr(const qtensor3<Tm>& qt3a, 
				 const qtensor3<Tm>& qt3b){
   assert(qt3a.dir == qt3b.dir); // bra dir fliped implicitly
   assert(qt3a.qmid == qt3b.qmid);
   assert(qt3a.qcol == qt3b.qcol);
   qsym sym = -qt3a.sym + qt3b.sym;
   qtensor2<Tm> qt2(sym, qt3a.qrow, qt3b.qrow);
   // loop over external indices
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk = qt2(br,bc);
	 if(blk.size() == 0) continue;
	 // loop over contracted indices
	 for(int bm=0; bm<qt3a.mids(); bm++){
	    for(int bx=0; bx<qt3a.cols(); bx++){
	       const auto& blka = qt3a(bm,br,bx);
	       const auto& blkb = qt3b(bm,bc,bx);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int im=0; im<qt3a.qmid.get_dim(bm); im++){
	          blk += linalg::xgemm("N","T",blka[im].conj(),blkb[im]); 
	       } // im	       
	    } // bx
	 } // bm
      } // bc
   } // br
   return qt2;
}

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
template <typename Tm>
qtensor2<Tm> contract_qt3_qt3_lc(const qtensor3<Tm>& qt3a, 
				 const qtensor3<Tm>& qt3b){
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   assert(qt3a.qrow == qt3b.qrow);
   assert(qt3a.qmid == qt3b.qmid);
   qsym sym = -qt3a.sym + qt3b.sym;
   qtensor2<Tm> qt2(sym, qt3a.qcol, qt3b.qcol); 
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
qtensor2<Tm> contract_qt3_qt3_lr(const qtensor3<Tm>& qt3a, 
				 const qtensor3<Tm>& qt3b){
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   assert(qt3a.qrow == qt3b.qrow);
   assert(qt3a.qcol == qt3b.qcol);
   qsym sym = -qt3a.sym + qt3b.sym;
   qtensor2<Tm> qt2(sym, qt3a.qmid, qt3b.qmid);
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

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
//  r--*--c
template <typename Tm>
qtensor3<Tm> contract_qt3_qt2_c(const qtensor3<Tm>& qt3a, 
			 	const qtensor2<Tm>& qt2b){
   assert(qt3a.dir[0] == !qt2b.dir[1]);
   assert(qt3a.qmid == qt2b.qcol);
   qsym sym = qt3a.sym + qt2b.sym;
   std::vector<bool> dir = {qt2b.dir[0], qt3a.dir[1], qt3a.dir[2]};
   qtensor3<Tm> qt3(sym, qt2b.qrow, qt3a.qrow, qt3a.qcol, dir);
   // loop over external indices
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.qmid.get_dim(bm);
      for(int br=0; br<qt3.rows(); br++){
         for(int bc=0; bc<qt3.cols(); bc++){
	    auto& blk = qt3(bm,br,bc);
	    if(blk.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.mids(); bx++){
	       const auto& blka = qt3a(bx,br,bc);
	       const auto& blkb = qt2b(bm,bx);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       int xdim = qt3a.qmid.get_dim(bx);
	       for(int x=0; x<xdim; x++){
	          for(int m=0; m<mdim; m++){
	             blk[m] += blkb(m,x)*blka[x];
	          } // m
	       } // x 
	    } // bx
	 } // bc
      } // br
   } // bm
   return qt3;
}

//  r/	m 
//   *  |     = [m](r,c) = op(r,x) A[m](x,c) = <mr|o|c>
//  x\--*--c
template <typename Tm>
qtensor3<Tm> contract_qt3_qt2_l(const qtensor3<Tm>& qt3a, 
				const qtensor2<Tm>& qt2b){
   assert(qt3a.dir[1] == !qt2b.dir[1]);
   assert(qt3a.qrow == qt2b.qcol);
   qsym sym = qt3a.sym + qt2b.sym;
   std::vector<bool> dir = {qt3a.dir[0], qt2b.dir[0], qt3a.dir[2]};
   qtensor3<Tm> qt3(sym, qt3a.qmid, qt2b.qrow, qt3a.qcol, dir);
   // loop over external indices
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.qmid.get_dim(bm);
      for(int br=0; br<qt3.rows(); br++){
         for(int bc=0; bc<qt3.cols(); bc++){
	    auto& blk = qt3(bm,br,bc);
	    if(blk.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.rows(); bx++){
	       const auto& blka = qt3a(bm,bx,bc);
	       const auto& blkb = qt2b(br,bx);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += linalg::xgemm("N","N",blkb,blka[m]);
	       } // m
	    } // bx
	 } // bc
      } // br
   } // bm
   return qt3;
}

//     m  \c/r
//     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
//  r--*--/x/c
template <typename Tm>
qtensor3<Tm> contract_qt3_qt2_r(const qtensor3<Tm>& qt3a, 
				const qtensor2<Tm>& qt2b){
   assert(qt3a.dir[2] == !qt2b.dir[1]); // each line is associated with one dir
   assert(qt3a.qcol == qt2b.qcol);
   qsym sym = qt3a.sym + qt2b.sym;
   std::vector<bool> dir = {qt3a.dir[0], qt3a.dir[1], qt2b.dir[0]};
   qtensor3<Tm> qt3(sym, qt3a.qmid, qt3a.qrow, qt2b.qrow, dir);
   // loop over external indices
   for(int bm=0; bm<qt3.mids(); bm++){
      int mdim = qt3.qmid.get_dim(bm);
      for(int br=0; br<qt3.rows(); br++){
         for(int bc=0; bc<qt3.cols(); bc++){
	    auto& blk = qt3(bm,br,bc);
	    if(blk.size() == 0) continue;
	    // loop over contracted indices
	    for(int bx=0; bx<qt3a.cols(); bx++){
	       const auto& blka = qt3a(bm,br,bx);
	       const auto& blkb = qt2b(bc,bx);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += linalg::xgemm("N","T",blka[m],blkb);
	       } // m
	    } // bx
	 } // bc
      } // br
   } // bm
   return qt3;
}

} // ctns

#endif
