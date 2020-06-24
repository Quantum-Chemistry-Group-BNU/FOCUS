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
   assert(qt2a.dir[1] == !qt2b.dir[0]);
   assert(qt2a.get_dim_col() == qt2b.get_dim_row());
   vector<bool> dir = {qt2a.dir[0], qt2b.dir[1]};
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
	    blk += xgemm("N","N",blka,blkb);
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
   assert(qt3a.dir[1] == !qt2b.dir[1] );
   assert(qt3a.get_dim_row() == qt2b.get_dim_col());
   vector<bool> dir = {qt3a.dir[0], qt2b.dir[0], qt3a.dir[2]};
   qtensor3 qt3(sym, qt3a.qmid, qt2b.qrow, qt3a.qcol, dir);
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
	          blk[m] += xgemm("N","N",blkb,blka[m]);
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
   assert(qt3a.dir[0] == !qt2b.dir[1]);
   assert(qt3a.get_dim_mid() == qt2b.get_dim_col());
   vector<bool> dir = {qt2b.dir[0], qt3a.dir[1], qt3a.dir[2]};
   qtensor3 qt3(sym, qt2b.qrow, qt3a.qrow, qt3a.qcol, dir);
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

//     m  
//     |    	= [m](r,c) = A[m](r,x)*op(x,c) [normal contraction] 
//  r--*--x-*-c 
qtensor3 tns::contract_qt3_qt2_r0(const qtensor3& qt3a, 
				  const qtensor2& qt2b){
   return contract_qt3_qt2_r(qt3a,qt2b.P()); 
}

//     m  \c/r
//     |  *  = [m](r,c) = A[m](r,x)*op(c,x) [permuted contraction (AO^T)]
//  r--*--/x/c
qtensor3 tns::contract_qt3_qt2_r(const qtensor3& qt3a, 
				 const qtensor2& qt2b){
   qsym sym = qt3a.sym + qt2b.sym;
   assert(qt3a.dir[2] == !qt2b.dir[1]); // each line is associated with one dir
   assert(qt3a.get_dim_col() == qt2b.get_dim_col());
   vector<bool> dir = {qt3a.dir[0], qt3a.dir[1], qt2b.dir[0]};
   qtensor3 qt3(sym, qt3a.qmid, qt3a.qrow, qt2b.qrow, dir);
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
	          blk[m] += xgemm("N","T",blka[m],blkb);
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
   assert(qt3a.get_dim_row() == qt3b.get_dim_row());
   assert(qt3a.get_dim_mid() == qt3b.get_dim_mid());
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   qsym sym = -qt3a.sym + qt3b.sym;
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
	          blk += xgemm("T","N",blka[m],blkb[m]); 
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
   assert(qt3a.get_dim_mid() == qt3b.get_dim_mid());
   assert(qt3a.get_dim_col() == qt3b.get_dim_col());
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   qsym sym = -qt3a.sym + qt3b.sym;
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
	          blk += xgemm("N","T",blka[m],blkb[m]); 
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
   assert(qt3a.get_dim_row() == qt3b.get_dim_row());
   assert(qt3a.get_dim_col() == qt3b.get_dim_col());
   assert(qt3a.dir == qt3b.dir); // bra dir fliped
   qsym sym = -qt3a.sym + qt3b.sym;
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
	             blk(ir,ic) += xgemm("N","N",blka[ir],blkb[ic].T()).trace();
		  } // r 
	       } // c
	    } // qy
	 } // qx
	 //---------------------------------- 
      } // qc
   } // qr
   return qt2;
}

// used in sampling with RCF v[l]*R[n,l,r]=>w[n,r]
qtensor2 tns::contract_qt3_vec_l(const qtensor3& qt3a, 
			         const qsym& sym_l, 
			         const matrix<double>& vec_l){
   assert(qt3a.sym == qsym(0,0));
   qsym sym = qt3a.sym + sym_l;
   vector<bool> dir = {1,1};
   qtensor2 qt2(sym, qt3a.qmid, qt3a.qcol, dir);
   for(const auto& pm : qt3a.qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& pc : qt3a.qcol){ 
	 auto qc = pc.first;
	 int cdim = pc.second;
	 auto& blk = qt2.qblocks[make_pair(qm,qc)];
	 auto& blk0 = qt3a.qblocks.at(make_tuple(qm,sym_l,qc));
	 if(blk.size() == 0 || blk0.size() == 0) continue;
	 for(int m=0; m<mdim; m++){
	    auto mat = xgemm("N","N",vec_l,blk0[m]); // (1,c)
	    for(int c=0; c<cdim; c++){
	       blk(m,c) = mat(0,c);
	    }
         }
      }
   }
   return qt2;
}
