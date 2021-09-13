#ifndef TENSOR_LINALG_H
#define TENSOR_LINALG_H

#include "../../core/linalg.h"

namespace ctns{

/*
template <typename Tm> inline bool is_matrix(){ return false; }
template <typename Tm> inline bool is_matrix<linalg::matrix<Tm>>{ return true; }
template <typename Tm> inline bool is_matrix<dtensor2<Tm>>{ return true; }
*/

// C := alpha*op( A )*op( B ) + beta*C
template <typename Tm, typename Km> //, typename Xm, typename Ym>
void xgemm(const char* TRANSA, const char* TRANSB,
	   const Km alpha, const dtensor2<Tm>& A, const dtensor2<Tm>& B,
           const Km beta, dtensor2<Tm>& C){
   //static_assert(is_matrix<Xm>() && is_matrix<Ym>());
   int M, N, K;
   int LDA, LDB, LDC;
   // TRANS is c-type string (character array), input "N" not 'N' 
   char trans_A = toupper(TRANSA[0]); 
   char trans_B = toupper(TRANSB[0]);
   assert(trans_A == 'N' || trans_A == 'T' || trans_A == 'C');
   assert(trans_B == 'N' || trans_B == 'T' || trans_B == 'C');
   if(trans_A == 'N'){
      M = A.rows(); K = A.cols(); LDA = M; 
   }else{
      M = A.cols(); K = A.rows(); LDA = K; 
   }
   if(trans_B == 'N'){
      assert(K == B.rows());
      N = B.cols(); LDB = K;
   }else{
      assert(K == B.cols());
      N = B.rows(); LDB = N;
   }
   // we assume the exact match of matrix size in our applications
   assert(C.rows() == M && C.cols() == N);
   LDC = M;
   Tm talpha = static_cast<Tm>(alpha);
   Tm tbeta = static_cast<Tm>(beta); 
   linalg::xgemm(&trans_A, &trans_B, &M, &N, &K, &talpha, 
	         A.data(), &LDA, B.data(), &LDB, &tbeta, 
	         C.data(), &LDC);
}

} // ctns

#endif
