#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>
#include <algorithm>

using namespace std;
using namespace linalg;
using namespace tns;

// --- tensor linear algebra : contractions ---

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

// --- symmetry operations : merge & expand operations ---

// one-dot wavefunction 
// matrix storage order : (lc,r) [fortran]  
qtensor2 tns::merge_qt3_qt2_lc(const qtensor3& qt3,
			       const qsym_space& qlc, 
			       const qsym_dpt& dpt){
   const auto& qcol = qt3.qcol;
   qtensor2 qt2(qt3.sym, qlc, qcol, {0,1,1});
   for(const auto& pc : qcol){
      auto qc = pc.first;
      int cdim = pc.second;
      for(const auto& plc : qlc){
         auto qcomb = plc.first;
	 for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;
	    auto qm = q12.second;
	    int rdim = get<0>(p12.second);
	    int mdim = get<1>(p12.second);
	    int ioff = get<2>(p12.second);
            auto& blk2 = qt2.qblocks[make_pair(qcomb,qc)];
	    const auto& blk3 = qt3.qblocks.at(make_tuple(qm,qr,qc));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
		        int imr = ioff+im*rdim+ir;
		        blk2(imr,ic) = blk3[im](ir,ic);
		     } // ir
		  } // ic
	       } // im
	    }
	 } // q12
      } // plc
   } // col
   return qt2;
}

qtensor3 tns::split_qt3_qt2_lc(const qtensor2& qt2,
			       const qsym_space& qlx,
			       const qsym_space& qcx,
			       const qsym_dpt& dpt){
   const auto& qlc  = qt2.qrow; 
   const auto& qcol = qt2.qcol;
   qtensor3 qt3(qt2.sym, qcx, qlx, qcol, {0,1,1,1});
   for(const auto& pc : qcol){
      auto qc = pc.first;
      int cdim = pc.second;
      for(const auto& plc : qlc){
         auto qcomb = plc.first;
	 for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;
	    auto qm = q12.second;
	    int rdim = get<0>(p12.second);
	    int mdim = get<1>(p12.second);
	    int ioff = get<2>(p12.second);
	    auto& blk3 = qt3.qblocks[make_tuple(qm,qr,qc)];
            const auto& blk2 = qt2.qblocks.at(make_pair(qcomb,qc));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
		        int imr = ioff+im*rdim+ir;
		        blk3[im](ir,ic) = blk2(imr,ic); 
		     } // ir
		  } // ic
	       } // im
	    }
	 } // q12
      } // plc
   } // col
   return qt3;
}

// matrix storage order : (l,rc) 
qtensor2 tns::merge_qt3_qt2_cr(const qtensor3& qt3,
			       const qsym_space& qcr, 
			       const qsym_dpt& dpt){;
   const auto& qrow = qt3.qrow;
   qtensor2 qt2(qt3.sym, qrow, qcr, {0,1,1});
   for(const auto& pcr : qcr){
      auto qcomb = pcr.first;
      for(const auto& p12 : dpt.at(qcomb)){
	 auto q12 = p12.first;
	 auto qm = q12.first;
	 auto qc = q12.second;
	 int mdim = get<0>(p12.second);
	 int cdim = get<1>(p12.second);
	 int ioff = get<2>(p12.second);
 	 for(const auto& pr : qrow){
	    auto qr = pr.first;
	    int rdim = pr.second;
            auto& blk2 = qt2.qblocks[make_pair(qr,qcomb)];
	    const auto& blk3 = qt3.qblocks.at(make_tuple(qm,qr,qc));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     int imc = ioff+im*cdim+ic;
		     for(int ir=0; ir<rdim; ir++){
		        blk2(ir,imc) = blk3[im](ir,ic);
		     } // ir
		  } // ic
	       } // im
	    }
	 } // row
      } // p12
   } // pcr
   return qt2;
}

qtensor3 tns::split_qt3_qt2_cr(const qtensor2& qt2,
			       const qsym_space& qcx,
			       const qsym_space& qrx,
			       const qsym_dpt& dpt){
   const auto& qrow = qt2.qrow;
   const auto& qcr  = qt2.qcol;
   qtensor3 qt3(qt2.sym, qcx, qrow, qrx, {0,1,1,1});
   for(const auto& pcr : qcr){
      auto qcomb = pcr.first;
      for(const auto& p12 : dpt.at(qcomb)){
	 auto q12 = p12.first;
	 auto qm = q12.first;
	 auto qc = q12.second;
	 int mdim = get<0>(p12.second);
	 int cdim = get<1>(p12.second);
	 int ioff = get<2>(p12.second);
 	 for(const auto& pr : qrow){
	    auto qr = pr.first;
	    int rdim = pr.second;
	    auto& blk3 = qt3.qblocks[make_tuple(qm,qr,qc)];
            const auto& blk2 = qt2.qblocks.at(make_pair(qr,qcomb));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     int imc = ioff+im*cdim+ic;
		     for(int ir=0; ir<rdim; ir++){
		        blk3[im](ir,ic) = blk2(ir,imc); 
		     } // ir
		  } // ic
	       } // im
	    }
	 } // row
      } // p12
   } // pcr
   return qt3;
}

// matrix storage order : (lr,c) 
qtensor2 tns::merge_qt3_qt2_lr(const qtensor3& qt3,
			       const qsym_space& qlr, 
			       const qsym_dpt& dpt){
   const auto& qmid = qt3.qmid;
   qtensor2 qt2(qt3.sym, qlr, qmid, {0,1,1});
   for(const auto& pm : qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& plr : qlr){
         auto qcomb = plr.first;
	 for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;
	    auto qc = q12.second;
	    int rdim = get<0>(p12.second); 
	    int cdim = get<1>(p12.second); 
	    int ioff = get<2>(p12.second);
	    auto& blk2 = qt2.qblocks[make_pair(qcomb,qm)];
	    const auto& blk3 = qt3.qblocks.at(make_tuple(qm,qr,qc));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
		        int icr = ioff+ic*rdim+ir;
		        blk2(icr,im) = blk3[im](ir,ic);
		     } // ir
		  } // ic
	       } // im
	    } 
	 }
      }
   }
   return qt2;
}

qtensor3 tns::split_qt3_qt2_lr(const qtensor2& qt2,
			       const qsym_space& qlx,
			       const qsym_space& qrx,
			       const qsym_dpt& dpt){
   const auto& qlr  = qt2.qrow;
   const auto& qmid = qt2.qcol;
   qtensor3 qt3(qt2.sym, qmid, qlx, qrx, {0,1,1,1});
   for(const auto& pm : qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& plr : qlr){
         auto qcomb = plr.first;
	 for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;
	    auto qc = q12.second;
	    int rdim = get<0>(p12.second); 
	    int cdim = get<1>(p12.second); 
	    int ioff = get<2>(p12.second);
	    auto& blk3 = qt3.qblocks[make_tuple(qm,qr,qc)];
	    const auto& blk2 = qt2.qblocks.at(make_pair(qcomb,qm));
	    if(blk2.size()>0 && blk3.size()>0){
	       for(int im=0; im<mdim; im++){
	          for(int ic=0; ic<cdim; ic++){
		     for(int ir=0; ir<rdim; ir++){
		        int icr = ioff+ic*rdim+ir;
		       	blk3[im](ir,ic) = blk2(icr,im);
		     } // ir
		  } // ic
	       } // im
	    } 
	 }
      }
   }
   return qt3;
}

// two-dot wavefunction
qtensor2 tns::merge_qt4_qt2_lr_c1c2(const qtensor4& qt4,
			            const qsym_space& qlr,
		                    const qsym_space& qc1c2,
			            const qsym_dpt& dpt1,
			            const qsym_dpt& dpt2){
   const auto& sym = qt4.sym;
   qtensor2 qt2(sym, qlr, qc1c2, {0, 1, 1});
   // col
   for(const auto& pc1c2 : qc1c2){
      auto qcomb2 = pc1c2.first;
      for(const auto& pcc12 : dpt2.at(qcomb2)){
	 auto qcc12 = pcc12.first;
	 auto qm = qcc12.first;
	 auto qv = qcc12.second;
         int mdim = get<0>(pcc12.second);
         int vdim = get<0>(pcc12.second);
         int joff = get<0>(pcc12.second);
	 // row
	 for(const auto& plr : qlr){
            auto qcomb1 = plr.first;
	    for(const auto& plr12 : dpt1.at(qcomb1)){
	       auto qlr12 = plr12.first;
	       auto qr = qlr12.first;
	       auto qc = qlr12.second;
	       int rdim = get<0>(plr12.second);
	       int cdim = get<0>(plr12.second);
	       int ioff = get<0>(plr12.second);
	       auto& blk2 = qt2.qblocks[make_pair(qcomb1,qcomb2)];
	       const auto& blk4 = qt4.qblocks.at(make_tuple(qm,qv,qr,qc));
	       if(blk2.size()>0 && blk4.size()>0){
		  for(int imv=0; imv<mdim*vdim; imv++){
		     int j = joff+imv; // im*vdim+iv
		     for(int ic=0; ic<cdim; ic++){
		        for(int ir=0; ir<rdim; ir++){
			   int i = ioff+ic*rdim+ir;
			   blk2(i,j) = blk4[imv](ir,ic);
			} // ir
		     } // ic
		  } // imv
	       }
	    } // plr12
	 } // plr
      }  // pcc12
   } // pc1c2
   return qt2;
}	

// A[l,c1,c2,r]->A[lc1,c2,r]
qtensor3 tns::merge_qt4_qt3_lc1(const qtensor4& qt4,
			        const qsym_space& qlc1, 
			        const qsym_dpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qver = qt4.qver;
   const auto& qcol = qt4.qcol;
   qtensor3 qt3(sym, qver, qlc1, qcol, {0,1,1,1});
   for(const auto& pv : qver){
      auto qv = pv.first;
      int vdim = pv.second;
      for(const auto& plc1 : qlc1){
         auto qcomb = plc1.first;
         for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;  // l
	    auto qm = q12.second; // c1
	    int rdim = get<0>(p12.second);
	    int mdim = get<1>(p12.second);
            int ioff = get<2>(p12.second);
	    for(const auto& pc : qcol){
               auto qc = pc.first;
	       int cdim = pc.second;
	       auto& blk3 = qt3.qblocks[make_tuple(qv,qcomb,qc)];
	       const auto& blk4 = qt4.qblocks.at(make_tuple(qm,qv,qr,qc)); 
	       if(blk3.size()>0 && blk4.size()>0){
	          // merging block blk3[v](lcdim,cdim)
		  for(int im=0; im<mdim; im++){
		     for(int iv=0; iv<vdim; iv++){
	               for(int ic=0; ic<cdim; ic++){
		          for(int ir=0; ir<rdim; ir++){
			     int ilc = ioff+im*rdim+ir; // store (ir,im) 
		             blk3[iv](ilc,ic) = blk4[im*vdim+iv](ir,ic);
		          } // ir
		       } // ic
		     } // iv
		  } // im
	       }
	    } // c
	 } // rm
      } // lc1
   } // v
   return qt3;
}

// A[l,c1,c2,r]<-A[lc1,c2,r]
qtensor4 tns::split_qt4_qt3_lc1(const qtensor3& qt3,
			        const qsym_space& qlx,
			        const qsym_space& qc1, 
			        const qsym_dpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qlc1 = qt3.qrow;
   const auto& qver = qt3.qmid;
   const auto& qcol = qt3.qcol;
   qtensor4 qt4(sym, qc1, qver, qlx, qcol);
   for(const auto& pv : qver){
      auto qv = pv.first;
      int vdim = pv.second;
      for(const auto& plc1 : qlc1){
         auto qcomb = plc1.first;
         for(const auto& p12 : dpt.at(qcomb)){
	    auto q12 = p12.first;
	    auto qr = q12.first;  // l
	    auto qm = q12.second; // c1
	    int rdim = get<0>(p12.second);
	    int mdim = get<1>(p12.second);
            int ioff = get<2>(p12.second);
	    for(const auto& pc : qcol){
               auto qc = pc.first;
	       int cdim = pc.second;
	       auto& blk4 = qt4.qblocks[make_tuple(qm,qv,qr,qc)]; 
	       const auto& blk3 = qt3.qblocks.at(make_tuple(qv,qcomb,qc));
	       if(blk3.size()>0 && blk4.size()>0){
	          // merging block blk3[v](lcdim,cdim) reversed
		  for(int im=0; im<mdim; im++){
		     for(int iv=0; iv<vdim; iv++){
	               for(int ic=0; ic<cdim; ic++){
		          for(int ir=0; ir<rdim; ir++){
			     int ilc = ioff+im*rdim+ir; // storage order (ir,im) 
			     blk4[im*vdim+iv](ir,ic) = blk3[iv](ilc,ic);
		          } // ir
		       } // ic
		     } // iv
		  } // im
	       }
	    } // c
	 } // rm
      } // lc1
   } // v
   return qt4;
}

// A[l,c1,c2,r]->A[l,c1,c2r]
qtensor3 tns::merge_qt4_qt3_c2r(const qtensor4& qt4,
			        const qsym_space& qc2r, 
			        const qsym_dpt& dpt){
   const auto& sym = qt4.sym;
   const auto& qrow = qt4.qrow; 
   const auto& qmid = qt4.qmid;
   qtensor3 qt3(sym, qmid, qrow, qc2r, {0,1,1,1});
   for(const auto& pm : qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qrow){
         auto qr = pr.first;
	 int rdim = pr.second;
	 for(const auto& pc2r : qc2r){
	    auto qcomb = pc2r.first;
	    for(const auto& p12 : dpt.at(qcomb)){
	       auto q12 = p12.first;
	       auto qv = q12.first;
	       auto qc = q12.second;
	       int vdim = get<0>(p12.second);
	       int cdim = get<1>(p12.second);
	       int ioff = get<2>(p12.second);
	       auto& blk3 = qt3.qblocks[make_tuple(qm,qr,qcomb)];
	       const auto& blk4 = qt4.qblocks.at(make_tuple(qm,qv,qr,qc)); 
	       if(blk3.size()>0 && blk4.size()>0){
		  for(int im=0; im<mdim; im++){
		     for(int iv=0; iv<vdim; iv++){
	               for(int ic=0; ic<cdim; ic++){
		          for(int ir=0; ir<rdim; ir++){
			     int icr = ioff+iv*cdim+ic; // storage order (ic,iv) 
		             blk3[im](ir,icr) = blk4[im*vdim+iv](ir,ic);
		          } // ir
		       } // ic
		     } // iv
		  } // im
	       }
	    } // vc
	 } // c2r 
      } // qr
   } // qm
   return qt3;
}

// A[l,c1,c2,r]<-A[l,c1,c2r]
qtensor4 tns::split_qt4_qt3_c2r(const qtensor3& qt3,
			        const qsym_space& qc2,
			        const qsym_space& qrx, 
			        const qsym_dpt& dpt){
   const auto& sym = qt3.sym;
   const auto& qrow = qt3.qrow; 
   const auto& qmid = qt3.qmid;
   const auto& qc2r = qt3.qcol;
   qtensor4 qt4(sym, qmid, qc2, qrow, qrx);
   for(const auto& pm : qmid){
      auto qm = pm.first;
      int mdim = pm.second;
      for(const auto& pr : qrow){
         auto qr = pr.first;
	 int rdim = pr.second;
	 for(const auto& pc2r : qc2r){
	    auto qcomb = pc2r.first;
	    for(const auto& p12 : dpt.at(qcomb)){
	       auto q12 = p12.first;
	       auto qv = q12.first;
	       auto qc = q12.second;
	       int vdim = get<0>(p12.second);
	       int cdim = get<1>(p12.second);
	       int ioff = get<2>(p12.second);
	       auto& blk4 = qt4.qblocks[make_tuple(qm,qv,qr,qc)]; 
	       const auto& blk3 = qt3.qblocks.at(make_tuple(qm,qr,qcomb));
	       if(blk3.size()>0 && blk4.size()>0){
		  for(int im=0; im<mdim; im++){
		     for(int iv=0; iv<vdim; iv++){
	               for(int ic=0; ic<cdim; ic++){
		          for(int ir=0; ir<rdim; ir++){
			     int icr = ioff+iv*cdim+ic; // storage order (ic,iv) 
			     blk4[im*vdim+iv](ir,ic) = blk3[im](ir,icr);
		          } // ir
		       } // ic
		     } // iv
		  } // im
	       }
	    } // vc
	 } // c2r 
      } // qr
   } // qm
   return qt4;
}
