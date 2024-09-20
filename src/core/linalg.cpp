#include <cassert>
#include <memory>
#include <string>
#include <cctype>
#include "linalg.h"

using namespace std;
using namespace linalg;

// eigen-decomposition HU=Ue: order=0/1 small-large/large-small

// real symmetric A
void linalg::eig_solver(const matrix<double>& A, vector<double>& e, 
			matrix<double>& U, const MKL_INT order){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   MKL_INT n = A.rows(), lwork = -1, liwork=-1, iworkopt, info = 0;
   double workopt;
   U.resize(A.rows(), A.cols());
   U = (order == 0)? A : -A;
   dsyevd_("V","L",&n,U.data(),&n,e.data(),&workopt,&lwork,&iworkopt,&liwork,&info);
   lwork = static_cast<MKL_INT>(workopt);
   liwork = static_cast<MKL_INT>(iworkopt);
   unique_ptr<double[]> work(new double[lwork]);
   unique_ptr<MKL_INT[]> iwork(new MKL_INT[liwork]);
   dsyevd_("V","L",&n,U.data(),&n,e.data(),work.get(),&lwork,iwork.get(),&liwork,&info);
   if(order == 1){ transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info){
      cout << "eig[d] failed with info=" << info << endl;
      exit(1);
   }
}

// complex Hermitian A
void linalg::eig_solver(const matrix<complex<double>>& A, vector<double>& e, 
		        matrix<complex<double>>& U, const MKL_INT order){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   MKL_INT n = A.rows(), lwork = -1, liwork=-1, lrwork = -1, iworkopt, info = 0;
   complex<double> workopt;
   double rworkopt;
   U.resize(A.rows(), A.cols());
   U = (order == 0)? A : -A;
   zheevd_("V","L",&n,U.data(),&n,e.data(),&workopt,&lwork,&rworkopt,&lrwork,&iworkopt,&liwork,&info);
   lwork = static_cast<MKL_INT>(workopt.real());
   lrwork = static_cast<MKL_INT>(rworkopt);
   liwork = static_cast<MKL_INT>(iworkopt);
   unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
   unique_ptr<double[]> rwork(new double[lrwork]);
   unique_ptr<MKL_INT[]> iwork(new MKL_INT[liwork]);
   zheevd_("V","L",&n,U.data(),&n,e.data(),work.get(),&lwork,rwork.get(),&lrwork,iwork.get(),&liwork,&info);
   if(order == 1){ transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info){
      cout << "eig[z] failed with info=" << info << endl;
      exit(1);
   }
}

// singular value decomposition: 
// iop =  0 : JOBU=A, JOBVT=A: all
//     =  1 : JOBU=S, JOBVT=N: the first min(m,n) columns of U 
//     =  2 : JOBU=N, JOBVT=S: the first min(m,n) rows of Vt
//     =  3 : JOBU=S, JOBVT=S: both
//     = 10 : JOBZ=A - divide-and-conquer version
//     = 13 : JOBZ=S (default)

// real A = U*s*Vt
void linalg::svd_solver(const matrix<double>& A, vector<double>& s,
			matrix<double>& U, matrix<double>& Vt,
			const MKL_INT iop){
   MKL_INT m = A.rows(), n = A.cols(), r = min(m,n);
   MKL_INT lwork = -1, ldu = 1, ldvt = 1, info;
   char JOBU, JOBVT, JOBZ;
   matrix<double> Atmp(A);
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
      }else if(iop == 3){
         JOBU = 'S'; U.resize(m,r); ldu = m;
         JOBVT = 'S'; Vt.resize(r,n); ldvt = r;
      }
      double workopt;
      dgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,&info);
      lwork = static_cast<MKL_INT>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,&info);
      //fail sometimes on lenovo/mkl2020
      //std::cout << "U.shape=" << U.rows() << "," << U.cols() << std::endl;
      //auto diff = linalg::check_orthogonality(U);
   }else{
      if(iop == 10){
         JOBZ = 'A'; U.resize(m,m); ldu = m; Vt.resize(n,n); ldvt = n;
      }else if(iop == 13){
         JOBZ = 'S'; U.resize(m,r); ldu = m; Vt.resize(r,n); ldvt = r;
      }
      unique_ptr<MKL_INT[]> iwork(new MKL_INT[8*r]);
      double workopt;
      dgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,iwork.get(),&info);
      lwork = static_cast<MKL_INT>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,iwork.get(),&info);
   }
   if(info){
      cout << "svd[d] failed with info=" << info << " for iop=" << iop << endl;
      exit(1);
   }
}

// complex A = U*s*Vh
void linalg::svd_solver(const matrix<complex<double>>& A, vector<double>& s,
			matrix<complex<double>>& U, matrix<complex<double>>& Vt,
			const MKL_INT iop){
   MKL_INT m = A.rows(), n = A.cols(), r = min(m,n);
   MKL_INT lwork = -1, ldu = 1, ldvt = 1, info;
   char JOBU, JOBVT, JOBZ;
   matrix<complex<double>> Atmp(A);
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
      }else if(iop == 3){
         JOBU = 'S'; U.resize(m,r); ldu = m;
         JOBVT = 'S'; Vt.resize(r,n); ldvt = r;
      }
      complex<double> workopt;
      unique_ptr<double[]> rwork(new double[5*r]);
      zgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,rwork.get(),&info);
      lwork = static_cast<MKL_INT>(workopt.real());
      unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
      zgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,rwork.get(),&info);
   }else{
      if(iop == 10){
         JOBZ = 'A'; U.resize(m,m); ldu = m; Vt.resize(n,n); ldvt = n;
      }else if(iop == 13){
         JOBZ = 'S'; U.resize(m,r); ldu = m; Vt.resize(r,n); ldvt = r;
      }
      complex<double> workopt;
      // lrwork: https://www.hpc.nec/documents/sdk/SDK_NLC/UsersGuide/man/zgesdd.html
      MKL_INT mx = max(m,n), mn = min(m,n);
      MKL_INT lrwork = mn*max(5*mn + 7, 2*mx + 2*mn + 1); 
      unique_ptr<double[]> rwork(new double[lrwork]);
      unique_ptr<MKL_INT[]> iwork(new MKL_INT[8*r]);
      zgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,rwork.get(),iwork.get(),&info);
      lwork = static_cast<MKL_INT>(workopt.real());
      unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
      zgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,rwork.get(),iwork.get(),&info);
   }
   if(info){
      cout << "svd[z] failed with info=" << info << " for iop=" << iop << endl;
      exit(1);
   }
}
