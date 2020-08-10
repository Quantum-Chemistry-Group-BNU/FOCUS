#ifndef CTNS_QTENSOR_CONTRACT_H
#define CTNS_QTENSOR_CONTRACT_H

#include <vector>
#include <string>
#include <map>
#include "../core/matrix.h"
#include "ctns_qsym.h"
#include "ctns_qtensor3.h"

namespace ctns{

//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^* Bn^T
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
   for(int br=0; br<qt2.qrow.size(); br++){
      for(int bc=0; bc<qt2.qcol.size(); bc++){
         auto& blk = qt2(br,bc);
	 if(blk.size() == 0) continue;
	 // loop over contracted indices
	 for(int bm=0; bm<qt3a.qmid.size(); bm++){
	    for(int bx=0; bx<qt3a.qcol.size(); bx++){
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

} // ctns

#endif
