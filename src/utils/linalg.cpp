#include "linalg.h"
#include <cassert>
#include <memory>

using namespace std;

// C = alpha*A*B + beta*C
void linalg::dgemm(const char* TRANSA, const char* TRANSB,
	   	   const double alpha, const matrix& A, const matrix& B,
	   	   const double beta, matrix& C){
   int M, N, K;
   int LDA, LDB, LDC;
   if(*TRANSA == 'N'){
       M = A.rows(); K = A.cols(); LDA = M; 
   }else{
       M = A.cols(); K = A.rows(); LDA = K; 
   }
   if(*TRANSB == 'N'){
       assert(K == B.rows());
       N = B.cols(); LDB = K;
   }else{
       assert(K == B.cols());
       N = B.rows(); LDB = N;
   }
   LDC = M;
   ::dgemm_(TRANSA, TRANSB, &M, &N, &K, &alpha, 
	    A.data(), &LDA, B.data(), &LDB, &beta, 
	    C.data(), &LDC);
}

// eigenvalues
void linalg::eig(matrix& A, vector<double>& e){
   assert(A.rows() == A.cols());  
   assert(A.rows() == e.size()); 
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

// Fnorm
double linalg::Fnorm(const matrix& A){
    int mn = A.size(), incx = 1;
    return dnrm2_(&mn,A.data(),&incx);
}
