#ifndef TENSOR_LINALG_H
#define TENSOR_LINALG_H

#include "../../core/matrix.h"
#include "../../core/linalg.h"

namespace ctns{

// C := alpha*op( A )*op( B ) + beta*C
template <typename Tm, typename Km>
void xgemm(const char* TRANSA, const char* TRANSB,
	   const Km alpha, const linalg::BaseMatrix<Tm>& A, const linalg::BaseMatrix<Tm>& B,
           const Km beta, dtensor2<Tm>& C){
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

// C := alpha*op( A )*op( B )
template <typename Tm>
linalg::matrix<Tm> xgemm(const char* TRANSA, const char* TRANSB,
	                 const linalg::BaseMatrix<Tm>& A, const linalg::BaseMatrix<Tm>& B,
			 const Tm alpha=1.0){
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
   linalg::matrix<Tm> C(M,N);
   LDC = M;
   const Tm beta = 0.0;
   linalg::xgemm(&trans_A, &trans_B, &M, &N, &K, &alpha, 
	         A.data(), &LDA, B.data(), &LDB, &beta, 
	         C.data(), &LDC);
   return C;
}

} // ctns

#endif
