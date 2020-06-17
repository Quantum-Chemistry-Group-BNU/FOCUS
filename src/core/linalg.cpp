#include "linalg.h"
#include <cassert>
#include <memory>
#include <string>
#include <cctype>

using namespace std;
using namespace linalg;

// eigendecomposition HU=Ue: order=0/1 small-large/large-small
void linalg::eig_solver(const matrix<double>& A, vector<double>& e, 
			matrix<double>& U, const int order){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   int n = A.rows(), lwork = -1, liwork=-1, iworkopt, info;
   double workopt;
   U.resize(A.rows(), A.cols());
   U = (order == 0)? A : -A;
   dsyevd_("V","L",&n,U.data(),&n,e.data(),&workopt,&lwork,&iworkopt,&liwork,&info);
   lwork = static_cast<int>(workopt);
   liwork = static_cast<int>(iworkopt);
   unique_ptr<double[]> work(new double[lwork]);
   unique_ptr<int[]> iwork(new int[liwork]);
   dsyevd_("V","L",&n,U.data(),&n,e.data(),work.get(),&lwork,iwork.get(),&liwork,&info);
   if(order == 1){ transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info){
      cout << "eig[d] failed with info=" << info << endl;
      exit(1);
   }
}

void linalg::eig_solver(const matrix<complex<double>>& A, vector<double>& e, 
		        matrix<complex<double>>& U, const int order){
   assert(A.rows() == A.cols());  
   assert(A.rows() <= e.size()); // allow larger space used for e 
   int n = A.rows(), lwork = -1, liwork=-1, lrwork = -1, iworkopt, info;
   complex<double> workopt;
   double rworkopt;
   U.resize(A.rows(), A.cols());
   U = (order == 0)? A : -A;
   zheevd_("V","L",&n,U.data(),&n,e.data(),&workopt,&lwork,&rworkopt,&lrwork,&iworkopt,&liwork,&info);
   lwork = static_cast<int>(workopt.real());
   lrwork = static_cast<int>(rworkopt);
   liwork = static_cast<int>(iworkopt);
   unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
   unique_ptr<double[]> rwork(new double[lrwork]);
   unique_ptr<int[]> iwork(new int[liwork]);
   zheevd_("V","L",&n,U.data(),&n,e.data(),work.get(),&lwork,rwork.get(),&lrwork,iwork.get(),&liwork,&info);
   if(order == 1){ transform(e.begin(),e.end(),e.begin(),[](const double& x){ return -x; }); }
   if(info){
      cout << "eig[z] failed with info=" << info << endl;
      exit(1);
   }
}

// A = U*s*Vt
void linalg::svd_solver(const matrix<double>& A, vector<double>& s,
			matrix<double>& U, matrix<double>& Vt,
			const int iop){
   int m = A.rows(), n = A.cols(), r = min(m,n);
   int lwork = -1, ldu = 1, ldvt = 1, info;
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
      lwork = static_cast<int>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,&info);
   }else{
      if(iop == 10){
         JOBZ = 'A'; U.resize(m,m); ldu = m; Vt.resize(n,n); ldvt = n;
      }else if(iop == 13){
         JOBZ = 'S'; U.resize(m,r); ldu = m; Vt.resize(r,n); ldvt = r;
      }
      unique_ptr<int[]> iwork(new int[8*r]);
      double workopt;
      dgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,iwork.get(),&info);
      lwork = static_cast<int>(workopt);
      unique_ptr<double[]> work(new double[lwork]);
      dgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,iwork.get(),&info);
   }
   if(info){
      cout << "svd[d] failed with info=" << info << endl;
      exit(1);
   }
}

void linalg::svd_solver(const matrix<complex<double>>& A, vector<double>& s,
			matrix<complex<double>>& U, matrix<complex<double>>& Vt,
			const int iop){
   int m = A.rows(), n = A.cols(), r = min(m,n);
   int lwork = -1, ldu = 1, ldvt = 1, info;
   matrix<complex<double>> Atmp(A);
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
      }else if(iop == 3){
         JOBU = 'S'; U.resize(m,r); ldu = m;
         JOBVT = 'S'; Vt.resize(r,n); ldvt = r;
      }
      complex<double> workopt;
      unique_ptr<double[]> rwork(new double[5*r]);
      zgesvd_(&JOBU,&JOBVT,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,rwork.get(),&info);
      lwork = static_cast<int>(workopt.real());
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
      int mx = max(m,n), mn = min(m,n), lrwork;
      if(mx > mn){
	 lrwork = 5*mn*mn + 5*mn;
      }else{
         lrwork = max(5*mn*mn + 5*mn, 2*mx*mn + 2*mn*mn + mn); 
      }
      unique_ptr<double[]> rwork(new double[lrwork]);
      unique_ptr<int[]> iwork(new int[8*r]);
      zgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,&workopt,&lwork,rwork.get(),iwork.get(),&info);
      lwork = static_cast<int>(workopt.real());
      unique_ptr<complex<double>[]> work(new complex<double>[lwork]);
      zgesdd_(&JOBZ,&m,&n,Atmp.data(),&m,s.data(),U.data(),&ldu,
              Vt.data(),&ldvt,work.get(),&lwork,rwork.get(),iwork.get(),&info);
   }
   if(info){
      cout << "svd[z] failed with info=" << info << endl;
      exit(1);
   }
}
