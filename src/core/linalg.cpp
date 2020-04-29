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
matrix linalg::dgemm(const char* TRANSA, const char* TRANSB,
		     const matrix& A, const matrix& B,
		     const double alpha){
   int M = (toupper(TRANSA[0]) == 'N')? A.rows() : A.cols(); 
   int N = (toupper(TRANSB[0]) == 'N')? B.cols() : B.rows(); 
   matrix C(M,N);
   dgemm(TRANSA,TRANSB,alpha,A,B,0.0,C);
   return C;
}

// eigenvalues: HC=CE
// order=0 from small to large; 
// order=1 from large to small.
void linalg::eigen_solver(matrix& A, vector<double>& e, const int order){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   int n = A.rows();
   int lwork = -1, liwork=-1;
   int iworkopt,info;
   double workopt;
   if(order == 1) A = -A;
   dsyevd_("V","L",&n,A.data(),&n,e.data(),&workopt,&lwork,
                   &iworkopt,&liwork,&info);
   lwork = static_cast<int>(workopt);
   liwork = static_cast<int>(iworkopt);
   unique_ptr<double[]> work(new double[lwork]);
   unique_ptr<int[]> iwork(new int[liwork]);
   dsyevd_("V","L",&n,A.data(),&n,e.data(),work.get(),&lwork,
                   iwork.get(),&liwork,&info);
   if(order == 1){
      transform(e.begin(),e.end(),e.begin(),
		[](const double& x){ return -x; });
   }
   if(info){
      cout << "dsyevd failed with info=" << info << endl;
      exit(1);
   }
}

// A = U*s*Vt
void linalg::svd_solver(matrix& A, vector<double>& s,
			matrix& U, matrix& Vt,
			const int iop){
   int m = A.rows(), n = A.cols(), r = min(m,n);
   int lwork = -1, ldu = 1, ldvt = 1, info;
   char JOBU, JOBVT, JOBZ;
   s.resize(r);
   if(iop < 10){
      if(iop == 0){
         JOBU = 'A'; U.resize(m,m); ldu = m;
         JOBVT = 'A'; Vt.resize(n,n); ldvt = n;
      }else if(iop == 1){
         JOBU = 'S'; U.resize(m,r); ldu = m;
         JOBVT = 'N';
      }else if(iop == 2){
         JOBU = 'N';
         JOBVT = 'S'; Vt.resize(r,n); ldvt = r;
      }
      double workopt;
      dgesvd_(&JOBU,&JOBVT,&m,&n,A.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,&info);
      lwork = static_cast<int>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesvd_(&JOBU,&JOBVT,&m,&n,A.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,&info);
   }else{
      if(iop == 10){
         JOBZ = 'A'; 
	 U.resize(m,m); ldu = m;
         Vt.resize(n,n); ldvt = n;
      }else if(iop == 11){
         JOBZ = 'S'; 
	 U.resize(m,r); ldu = m;
	 Vt.resize(r,n); ldvt = r;
      }
      unique_ptr<int[]> iwork(new int[8*r]);
      double workopt;
      dgesdd_(&JOBZ,&m,&n,A.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,iwork.get(),&info);
      lwork = static_cast<int>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesdd_(&JOBZ,&m,&n,A.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,iwork.get(),&info);
   }
   if(info){
      cout << "svd failed with info=" << info << endl;
      exit(1);
   }
}

// norm_F = sqrt(\sum_{ij}|aij|^2)
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

// ||A - At||_F
double linalg::symmetric_diff(const matrix& A){
   assert(A.rows() == A.cols());
   matrix At(A);
   At -= A.transpose();
   return normF(At);
}
