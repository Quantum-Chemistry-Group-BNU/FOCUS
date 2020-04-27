#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-2 tensor ---
void qtensor2::print(const string msg, const int level) const{
   cout << "qtensor2: " << msg << " msym=" << msym << endl;
   qsym_space_print(qrow,"qrow");
   qsym_space_print(qcol,"qcol");
   if(level >= 1){
      cout << "qblocks: nblocks=" << qblocks.size() << endl;
      int nnz = 0;
      for(const auto& p : qblocks){
         auto& t = p.first;
         auto& m = p.second;
         auto sym_row = get<0>(t);
         auto sym_col = get<1>(t);
         if(m.size() > 0){
            nnz++;
            cout << "idx=" << nnz 
		 << " block[" << sym_row << "," << sym_col << "]"
                 << " size=" << m.size() 
                 << endl; 
            if(level >= 2){
               m.print("mat");
            } // level=2
         }
      }
      cout << "total no. of nonzero blocks=" << nnz << endl;
   } // level=1
}

matrix qtensor2::to_matrix() const{
   int m = get_dim_row();
   int n = get_dim_col();
   matrix mat(m,n);
   int joff = 0;
   for(const auto& pj : qcol){
      const auto& qsymj = pj.first;
      const int mj = pj.second;
      int ioff = 0;
      for(const auto& pi : qrow){
	 const auto& qsymi = pi.first;
	 const int mi = pi.second;
	 const auto& blk = qblocks.at(make_pair(qsymi,qsymj));
	 if(blk.size() != 0){
	    // save
	    for(int j=0; j<mj; j++){
	       for(int i=0; i<mi; i++){
	          mat(ioff+i,joff+j) = blk(i,j);
	       } 
	    }
	 }
	 ioff += mi; 
      } // pi
      joff += mj;
   } // pj
   return mat;
}

// --- rank-3 tensor ---
void qtensor3::print(const string msg, const int level) const{
   cout << "qtensor3: " << msg << endl;
   qsym_space_print(qmid,"qmid");
   qsym_space_print(qrow,"qrow");
   qsym_space_print(qcol,"qcol");
   if(level >= 1){
      cout << "qblocks: nblocks=" << qblocks.size() << endl;
      int nnz = 0;
      for(const auto& p : qblocks){
         auto& t = p.first;
         auto& m = p.second;
         auto sym_mid = get<0>(t);
         auto sym_row = get<1>(t);
         auto sym_col = get<2>(t);
         if(m.size() > 0){
            nnz++;
            cout << "idx=" << nnz 
		 << " block[" << sym_mid << "," << sym_row << "," << sym_col << "]"
                 << " size=" << m.size() 
                 << " rows,cols=(" << m[0].rows() << "," << m[0].cols() << ")" 
                 << endl; 
            if(level >= 2){
               for(int i=0; i<m.size(); i++){		 
                  m[i].print("mat"+to_string(i));
               }
            } // level=2
         }
      }
      cout << "total no. of nonzero blocks=" << nnz << endl;
   } // level=1
}

// --- tensor linear algebra : contractions ---

//          r---*--\ qt3a
// q(r,c) =     |m |x     = <r|c> = \sum_n An^C*Bn^T
//          c---*--/ qt3b
qtensor2 tns::contract_qt3_qt3_cr(const qtensor3& qt3a, const qtensor3& qt3b){
   qtensor2 qt2;
   qt2.qrow = qt3a.qrow;
   qt2.qcol = qt3b.qrow;
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 // loop over contracted indices
	 vector<pair<qsym,qsym>> ilist;
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_tuple(msym,csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(make_pair(msym,xsym));
	    } // qx
	 } // qm
	 // perform contractions
	 auto key = make_pair(rsym,csym);
	 if(ilist.size() == 0){
	    qt2.qblocks[key] = matrix();
	 }else{
	    matrix mat(rdim,cdim);
	    for(const auto& p : ilist){
	       const qsym& msym = p.first;
	       const qsym& xsym = p.second;
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_tuple(msym,csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       int mdim = qt3b.qmid.at(msym);
               for(int m=0; m<mdim; m++){
	          mat += dgemm("N","T",blka[m],blkb[m]); 
	       } // m
	    } // qm,qx
	    qt2.qblocks[key] = mat;
	 }
      } // qc
   } // qr
   return qt2;
}

//     m  \r2
//     |  *   = [m](r,r2) = A[m](r,x)*R(r2,x)^T
//  r--*--/x 
qtensor3 tns::contract_qt3_qt2_r(const qtensor3& qt3a, const qtensor2& qt2b){
   qtensor3 qt3;
   qt3.qmid = qt3a.qmid;
   qt3.qrow = qt3a.qrow;
   qt3.qcol = qt2b.qrow;
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
	    // loop over contracted indices
	    vector<qsym> ilist;
	    for(const auto& px : qt3a.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_pair(csym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(xsym);
	    } // qx
	    // perform contractions
	    auto key = make_tuple(msym,rsym,csym);
	    if(ilist.size() == 0){
	       qt3.qblocks[key] = empty_block;
	    }else{
	       for(int m=0; m<mdim; m++){
	          matrix mat(rdim,cdim);
		  for(const auto& xsym : ilist){
	             auto keya = make_tuple(msym,rsym,xsym);
	             auto keyb = make_pair(csym,xsym);
	             auto& blka = qt3a.qblocks.at(keya);
	             auto& blkb = qt2b.qblocks.at(keyb);
		     mat += dgemm("N","T",blka[m],blkb);
		  }
	          qt3.qblocks[key].push_back(mat);
	       } // m
	    } 
	 } // qm
      } // qc
   } // qr
   return qt3;
}

//     |m/r
//     *	 
//     |x/c  = [m](r,c) = B(m,x)* A[x](r,c)
//  r--*--c
qtensor3 tns::contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b){
   qtensor3 qt3;
   qt3.qmid = qt2b.qrow;
   qt3.qrow = qt3a.qrow;
   qt3.qcol = qt3a.qcol;
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
	    // loop over contracted indices
	    vector<qsym> ilist;
	    for(const auto& px : qt3a.qmid){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(xsym,rsym,csym);
	       auto keyb = make_pair(msym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(xsym);
	    } // qx
	    // perform contractions
	    auto key = make_tuple(msym,rsym,csym);
	    if(ilist.size() == 0){
	       qt3.qblocks[key] = empty_block;
	    }else{
	       for(int m=0; m<mdim; m++){
	          matrix mat(rdim,cdim);
		  for(const auto& xsym : ilist){
	             auto keya = make_tuple(xsym,rsym,csym);
	             auto keyb = make_pair(msym,xsym);
	             auto& blka = qt3a.qblocks.at(keya);
	             auto& blkb = qt2b.qblocks.at(keyb);
		     int xdim = blkb.rows();
		     for(int x=0; x<xdim; x++){
		        mat += blkb(m,x)*blka[x];
	 	     } // x 
		  } // qx
	          qt3.qblocks[key].push_back(mat);
	       } // m
	    } 
	 } // qm
      } // qc
   } // qr
   return qt3;
}

//          /--*--r qt3a
// q(r,c) = |x |m  	  = <r|c> = \sum_n An^H*Bn
//          \--*--c qt3b
qtensor2 tns::contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b){
   qtensor2 qt2;
   qt2.qrow = qt3a.qcol;
   qt2.qcol = qt3b.qcol;
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 // loop over contracted indices
	 vector<pair<qsym,qsym>> ilist;
         for(const auto& pm : qt3a.qmid){
            const qsym& msym = pm.first;
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,rsym);
	       auto keyb = make_tuple(msym,xsym,csym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(make_pair(msym,xsym));
	    } // qx
	 } // qm
	 // perform contractions
	 auto key = make_pair(rsym,csym);
	 if(ilist.size() == 0){
	    qt2.qblocks[key] = matrix();
	 }else{
	    matrix mat(rdim,cdim);
	    for(const auto& p : ilist){
	       const qsym& msym = p.first;
	       const qsym& xsym = p.second;
	       auto keya = make_tuple(msym,xsym,rsym);
	       auto keyb = make_tuple(msym,xsym,csym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       int mdim = qt3b.qmid.at(msym);
               for(int m=0; m<mdim; m++){
	          mat += dgemm("T","N",blka[m],blkb[m]); 
	       } // m
	    } // qm,qx
	    qt2.qblocks[key] = mat;
	 }
      } // qc
   } // qr
   return qt2;
}

//  r/	m 
//   *  |     = [m](r,c) = L(r,x)*A[m](x,c)
//  x\--*--c
qtensor3 tns::contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b){
   qtensor3 qt3;
   qt3.qmid = qt3a.qmid;
   qt3.qrow = qt2b.qrow;
   qt3.qcol = qt3a.qcol;
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
	    // loop over contracted indices
	    vector<qsym> ilist;
	    for(const auto& px : qt3a.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,xsym,csym);
	       auto keyb = make_pair(rsym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(xsym);
	    } // qx
	    // perform contractions
	    auto key = make_tuple(msym,rsym,csym);
	    if(ilist.size() == 0){
	       qt3.qblocks[key] = empty_block;
	    }else{
	       for(int m=0; m<mdim; m++){
	          matrix mat(rdim,cdim);
		  for(const auto& xsym : ilist){
	             auto keya = make_tuple(msym,xsym,csym);
	             auto keyb = make_pair(rsym,xsym);
	             auto& blka = qt3a.qblocks.at(keya);
	             auto& blkb = qt2b.qblocks.at(keyb);
		     mat += dgemm("N","N",blkb,blka[m]);
		  }
	          qt3.qblocks[key].push_back(mat);
	       } // m
	    } 
	 } // qm
      } // qc
   } // qr
   return qt3;
}
