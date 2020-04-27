#include "tns_qtensor.h"
#include "../core/linalg.h"
#include <iostream>

using namespace std;
using namespace linalg;
using namespace tns;

// --- rank-2 tensor ---
void qtensor2::print(const string msg, const int level){
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
void qtensor3::print(const string msg, const int level){
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

//          c---*--\ qt3a
// q(r,c) =     |m |x
//          r---*--/ qt3b
qtensor2 tns::contract_qt3_qt3_cr(const qtensor3& qt3a, const qtensor3& qt3b){
   qtensor2 qt2;
   qt2.qrow = qt3b.qrow;
   qt2.qcol = qt3a.qrow;
   // loop over external indices
   for(const auto& pr : qt2.qrow){
      const qsym& rsym = pr.first; 
      int rdim = pr.second;
      for(const auto& pc : qt2.qcol){
	 const qsym& csym = pc.first;
	 int cdim = pc.second;
	 // loop over contracted indices
	 vector<pair<qsym,qsym>> ilist;
         for(const auto& pm : qt3b.qmid){
            const qsym& msym = pm.first;
	    for(const auto& px : qt3b.qcol){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,csym,xsym);
	       auto keyb = make_tuple(msym,rsym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(make_pair(msym,xsym));
	    } // x
	 } // m
	 // perform contractions
	 auto key = make_pair(rsym,csym);
	 if(ilist.size() == 0){
	    qt2.qblocks[key] = matrix();
	 }else{
	    matrix mat(rdim,cdim);
	    for(const auto& p : ilist){
	       const qsym& msym = p.first;
	       const qsym& xsym = p.second;
	       auto keya = make_tuple(msym,csym,xsym);
	       auto keyb = make_tuple(msym,rsym,xsym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt3b.qblocks.at(keyb);
	       int mdim = qt3b.qmid.at(msym);
               for(int m=0; m<mdim; m++){
	          mat += dgemm("N","T",blkb[m],blka[m]); 
	       } // m
	    }
	    qt2.qblocks[key] = mat;
	 }
      } // c
   } // r
   return qt2;
}

//     m  \c
//     |  *
//  r--*--/x 
qtensor3 tns::contract_qt3_qt2_r(const qtensor3& qt3a, const qtensor2& qt2b){
   qtensor3 qt3;
   qt3.qmid = qt3a.qmid;
   qt3.qrow = qt3a.qrow;
   qt3.qcol = qt2b.qcol;
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
	    for(const auto& px : qt2b.qrow){
	       const qsym& xsym = px.first;
	       // contract blocks
	       auto keya = make_tuple(msym,rsym,xsym);
	       auto keyb = make_pair(xsym,csym);
	       auto& blka = qt3a.qblocks.at(keya);
	       auto& blkb = qt2b.qblocks.at(keyb);
	       if(blka.size() == 0 || blkb.size() == 0) continue;
	       ilist.push_back(xsym);
	    }
	    // perform contractions
	    auto key = make_tuple(msym,rsym,csym);
	    if(ilist.size() == 0){
	       qt3.qblocks[key] = empty_block;
	    }else{
	       for(int m=0; m<mdim; m++){
	          matrix mat(rdim,cdim);
		  for(const auto& xsym : ilist){
	             auto keya = make_tuple(msym,rsym,xsym);
	             auto keyb = make_pair(xsym,csym);
	             auto& blka = qt3a.qblocks.at(keya);
	             auto& blkb = qt2b.qblocks.at(keyb);
		     mat += dgemm("N","N",blka[m],blkb);
		  }
	          qt3.qblocks[key].push_back(mat);
	       } // m
	    } 
	 } // m
      } // c
   } // r
   return qt3;
}

//          /--*--r qt3a
// q(r,c) = |  |  
//          \--*--c qt3b
qtensor2 tns::contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b){

}

//  r/	  
//   *  |  
//   \--*--c
qtensor3 tns::contract_qt3_qt2_l(const qtensor3& qt3a, const qtensor2& qt2b){

}

//     |c
//     *	  
//     |r  
//   --*--
qtensor3 tns::contract_qt3_qt2_c(const qtensor3& qt3a, const qtensor2& qt2b){

}
