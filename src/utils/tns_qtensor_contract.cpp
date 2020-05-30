#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- tensor linear algebra : contractions ---

// xgemm : C[i,k] = A[i,j]*B[j,k]
qtensor2 tns::contract_qt2_qt2(const qtensor2& qt2a, 
			       const qtensor2& qt2b){
   qsym sym = qt2a.sym + qt2b.sym;
   assert(qt2a.dir[2] == !qt2b.dir[1]);
   std::vector<bool> dir = {0, qt2a.dir[1], qt2b.dir[2]};
   qtensor2 qt2(sym, qt2a.qrow, qt2b.qcol, dir); 
   for(const auto& pr : qt2.qrow){
      const qsym& qr = pr.first;
      for(const auto& pc : qt2.qcol){
	 const qsym& qc = pc.first;
	 auto key = make_pair(qr,qc);
	 auto& blk = qt2.qblocks[key];
	 if(blk.size() == 0) continue;
	 //---------------------------------- 
	 // loop over contracted indices
	 for(const auto& px : qt2a.qcol){
	    const qsym& qx = px.first;
	    // contract blocks
	    auto keya = make_pair(qr,qx);
	    auto keyb = make_pair(qx,qc);
	    auto& blka = qt2a.qblocks.at(keya);
	    auto& blkb = qt2b.qblocks.at(keyb);
	    if(blka.size() == 0 || blkb.size() == 0) continue;
	    blk += dgemm("N","N",blka,blkb);
	 } // qx
	 //---------------------------------- 
      } // qc
   } // qr
   return qt2;
}

//  r/	m 
//   *  |     = [m](r,c) = op(r,x)*A[m](x,c) = <mr|o|c>
//  x\--*--c
qtensor3 tns::contract_qt3_qt2_l(const qtensor3& qt3a, 
				 const qtensor2& qt2b){
   qsym sym = qt3a.sym + qt2b.sym;
   qtensor3 qt3(sym, qt3a.qmid, qt2b.qrow, qt3a.qcol, qt3a.dir);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;
	    //---------------------------------- 
	    // loop over contracted indices
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,csym);
	       auto keyb = make_pair(rsym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += dgemm("N","N",blkb,blka[m]);
	       } // m
	    } // qx
	    //---------------------------------- 
	 } // qc
      } // qr
   } // qm
   return qt3;
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x)* A[x](r,c)
//  r--*--c
qtensor3 tns::contract_qt3_qt2_c(const qtensor3& qt3a, 
			 	 const qtensor2& qt2b){
   qsym sym = qt3a.sym + qt2b.sym;
   qtensor3 qt3(sym, qt2b.qrow, qt3a.qrow, qt3a.qcol, qt3a.dir);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;
	    //---------------------------------- 
	    // loop over contracted indices
	    for(const auto& px : qt3a.qmid){
	       const qsym& xsym = px.first;
	       int xdim = px.second;
	       // contract blocks
	       auto keya = make_tuple(xsym,rsym,csym);
	       auto keyb = make_pair(msym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int x=0; x<xdim; x++){
	          for(int m=0; m<mdim; m++){
	             blk[m] += blkb(m,x)*blka[x];
	          } // m
	       } // x 
	    } // qx
	    //---------------------------------- 
	 } // qc
      } // qr
   } // qm
   return qt3;
}

//     m  \c
//     |  *  = [m](r,c) = A[m](r,x)*op(c,x) 
//  r--*--/x
qtensor3 tns::contract_qt3_qt2_r(const qtensor3& qt3a, 
				 const qtensor2& qt2b){
   qsym sym = qt3a.sym + qt2b.sym;
   qtensor3 qt3(sym, qt3a.qmid, qt3a.qrow, qt2b.qrow, qt3a.dir);
   // loop over external indices
   for(const auto& pm : qt3.qmid){
      const qsym& msym = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qt3.qrow){
         const qsym& rsym = pr.first; 
         int rdim = pr.second;
         for(const auto& pc : qt3.qcol){
            const qsym& csym = pc.first;
            int cdim = pc.second;
	    auto key = make_tuple(msym,rsym,csym);
	    auto& blk = qt3.qblocks[key];
	    if(blk.size() == 0) continue;
	    //---------------------------------- 
	    // loop over contracted indices
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_pair(csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int m=0; m<mdim; m++){
	          blk[m] += dgemm("N","T",blka[m],blkb);
	       } // m
	    } // qx
	    //---------------------------------- 
	 } // qc
      } // qr
   } // qm
   return qt3;
}

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
qtensor2 tns::contract_qt3_qt3_lc(const qtensor3& qt3a, 
				  const qtensor3& qt3b){
   qsym sym = qt3a.sym + qt3b.sym;
   qtensor2 qt2(sym, qt3a.qcol, qt3b.qcol); 
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(rsym,csym);
	 auto& blk = qt2.qblocks[key];
	 if(blk.size() == 0) continue;
	 //---------------------------------- 
	 // loop over contracted indices
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    int mdim = pm.second;
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,rsym);
	       auto keyb = make_tuple(msym,xsym,csym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
               for(int m=0; m<mdim; m++){
	          blk += dgemm("T","N",blka[m],blkb[m]); 
	       } // m
	    } // qx
	 } // qm
	 //---------------------------------- 
      } // qc
   } // qr
   return qt2;
}

//          r--*--\ qt3a
// q(r,c) =    |m |x	  = <r|c> = \sum_n An^**Bn^T
//          c--*--/ qt3b
qtensor2 tns::contract_qt3_qt3_cr(const qtensor3& qt3a, 
				  const qtensor3& qt3b){
   qsym sym = qt3a.sym + qt3b.sym;
   qtensor2 qt2(sym, qt3a.qrow, qt3b.qrow);
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(rsym,csym);
	 auto& blk = qt2.qblocks[key];
	 if(blk.size() == 0) continue;
	 //---------------------------------- 
	 // loop over contracted indices
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    int mdim = pm.second;
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_tuple(msym,csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
               for(int m=0; m<mdim; m++){
	          blk += dgemm("N","T",blka[m],blkb[m]); 
	       } // m
	    } // qx
	 } // qm
	 //---------------------------------- 
      } // qc
   } // qr
   return qt2;
}

// 	      r|
//          /--*--\ qt3a
// q(r,c) = |x    |y	  = <r|c> = tr(A[r]^**B[c]^T)
//          \--*--/ qt3b
//            c|
qtensor2 tns::contract_qt3_qt3_lr(const qtensor3& qt3a, 
				  const qtensor3& qt3b){
   qsym sym = qt3a.sym + qt3b.sym;
   qtensor2 qt2(sym, qt3a.qmid, qt3b.qmid);
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 auto key = make_pair(rsym,csym);
	 auto& blk = qt2.qblocks[key];
	 if(blk.size() == 0) continue;
	 //---------------------------------- 
	 // loop over contracted indices
         for(const auto& px : qt3a.qrow){
            const qsym& xsym = px.first;
	    for(const auto& py : qt3a.qcol){
	       const qsym& ysym = py.first;
	       // contract blocks
	       auto keya = make_tuple(rsym,xsym,ysym);
	       auto keyb = make_tuple(csym,xsym,ysym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
	             blk(ir,ic) += dgemm("N","N",blka[ir],blkb[ic].T()).trace();
		  } // r 
	       } // c
	    } // qy
	 } // qx
	 //---------------------------------- 
      } // qc
   } // qr
   return qt2;
}
