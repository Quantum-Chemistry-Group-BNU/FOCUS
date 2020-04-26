#include "tns_qtensor.h"

using namespace std;
using namespace tns;
using namespace linalg;

// --- rank-2 tensor ---
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
	 if(blk.size() == 0) continue;
	 // save
	 for(int j=0; j<mj; j++){
	    for(int i=0; i<mi; i++){
	       mat(ioff+i,joff+j) = blk(i,j);
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
      cout << "total no. of nonzero blocks =" << nnz << endl;
   } // level=1
}

// --- tensor linear algebra : contractions ---

//          c---*--\ qt3a
// q(r,c) =     |  |
//          r---*--/ qt3b
qtensor2 contract_qt3_qt3_cr(const qtensor3& qt3a, const qtensor3& qt3b){
   qtensor2 qt2;
   qt2.qrow = qt3b.qrow;
   qt2.qcol = qt3a.qcol;
}

//          /--*--r qt3a
// q(r,c) = |  |  
//          \--*--c qt3b
qtensor2 contract_qt3_qt3_lc(const qtensor3& qt3a, const qtensor3& qt3b){

}

//	  \c
//     |  *
//  r--*--/ 
qtensor3 contract_qt3_qt2_r(const qtensor3& qt3, const qtensor2& qt2){

}

//  r/	  
//   *  |  
//   \--*--c
qtensor3 contract_qt3_qt2_l(const qtensor3& qt3, const qtensor2& qt2){

}

//     |c
//     *	  
//     |r  
//   --*--
qtensor3 contract_qt3_qt2_c(const qtensor3& qt3, const qtensor2& qt2){

}
