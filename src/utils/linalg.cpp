#include "linalg.h"
#include <cassert>
#include <memory>
#include <string>
#include <cctype>

using namespace std;
using namespace linalg;

// C = alpha*A*B + beta*C
void linalg::dgemm(const char* TRANSA, const char* TRANSB,
	   	   const double alpha, const matrix& A, const matrix& B,
	   	   const double beta, matrix& C){
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
   assert(M == C.rows() && N == C.cols());
   LDC = M; // we assume the exact match of matrix size in our applications
   ::dgemm_(&trans_A, &trans_B, &M, &N, &K, &alpha, 
	    A.data(), &LDA, B.data(), &LDB, &beta, 
	    C.data(), &LDC);
}

// shorthand for A*B
matrix linalg::mdot(const matrix& A, const matrix& B){
   int M = A.rows(), N = B.cols();
   matrix C(M,N);
   dgemm("N","N",1.0,A,B,0.0,C);
   return C;
}

// eigenvalues: HC=CE
void linalg::eigen_solver(matrix& A, vector<double>& e){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   int n = A.rows();
   int lwork = -1, liwork=-1;
   int iworkopt,info;
   double workopt;
   dsyevd_("V","L",&n,A.data(),&n,e.data(),&workopt,&lwork,
                   &iworkopt,&liwork,&info);
   lwork = static_cast<int>(workopt);
   liwork = static_cast<int>(iworkopt);
   std::unique_ptr<double[]> work(new double[lwork]);
   std::unique_ptr<int[]> iwork(new int[liwork]);
   dsyevd_("V","L",&n,A.data(),&n,e.data(),work.get(),&lwork,
                   iwork.get(),&liwork,&info);
   if(info){
      std::cout << "dsyevd failed with info=" << info << std::endl;
      exit(1);
   }
}

// normF = sqrt(\sum_{ij}|aij|^2)
double linalg::normF(const matrix& A){
   int mn = A.size(), incx = 1;
   return dnrm2_(&mn,A.data(),&incx);
}

// dnrm2
double linalg::dnrm2(const int N, const double* X){
   int INCX = 1;
   return ::dnrm2_(&N, X, &INCX);
}

// ddot
double linalg::ddot(const int N, const double* X, const double* Y){
   int INCX = 1, INCY = 1;
   return ::ddot_(&N, X, &INCX, Y, &INCY);
}

// transpose
matrix linalg::transpose(const matrix& A){
   matrix At(A.cols(),A.rows());
   for(int j=0; j<At.cols(); j++){
      for(int i=0; i<At.rows(); i++){
	 At(i,j) = A(j,i);
      }
   }
   return At;
}

// ||A - At||F
double linalg::symmetric_diff(const matrix& A){
   assert(A.rows() == A.cols());
   matrix At(A);
   At -= transpose(A);
   return normF(At);
}
