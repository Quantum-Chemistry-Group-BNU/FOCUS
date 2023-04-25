#include <iostream>
#include "tools.h"
#include "matrix.h"
#include "linalg.h"
#include "tests_core.h"

using namespace std;
using namespace linalg;

int tests::test_linalg(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_linalg" << endl;
   cout << tools::line_separator << endl;
   

   const int n = 3;
   const double thresh = 1.e-10;

   // --- real matrix ---
   matrix<double> mat(n,n);
   mat(1,2) = 1.0;
   mat(2,0) = 3.0;
   mat.print("mat");
   cout << mat.normF() << endl;
   cout << mat.diff_hermitian() << endl;

   auto mat2 = xgemm("N","N",mat,mat);
   mat2.print("mat2");
   
   auto matx(mat);
   auto mat3 = xgemm("N","N",mat,matx);
   mat3.print("mat3");

   // --- complex matrix --- 
   const complex<double> i(0.0,1.0);
   matrix<complex<double>> cmat(n,n);
   cmat(1,2) = 1.0;
   cmat(2,0) = 3.0;
   cmat(2,1) = 1.0+1.0*i;
   cmat.print("cmat");
   cout << cmat.normF() << endl;
   cout << cmat.diff_hermitian() << endl;

   const complex<double> ctmp={1.0,0.0};
   cmat += ctmp*random_matrix<complex<double>>(n,n);
   auto cmat2 = xgemm("C","N",cmat,cmat);
   cmat2.print("cmat2");
   
   auto cmatx(cmat);
   auto cmat3 = xgemm("C","N",cmat,cmatx);
   cmat3.print("cmat3");

   // --- linalg : SVD --- 
   cout << "\ntest svd_solver" << endl;
   cout << "\nreal version:" << endl;
   vector<double> s;
   matrix<double> U, Vt;
   svd_solver(mat, s, U, Vt, 3);
   cout << "s0=" << s[0] << endl;
   auto tmp1 = xgemm("N","N",U,diagonal_matrix(s));
   auto tmp2 = xgemm("N","N",tmp1,Vt);
   auto diff_svd = (mat-tmp2).normF();
   cout << "diff_svd=" << diff_svd << endl;
   assert(diff_svd < thresh);
  
   cout << "\ncomplex version:" << endl;
   vector<double> sc;
   matrix<complex<double>> Uc, Vtc;
   svd_solver(cmat, sc, Uc, Vtc, 0);
   cout << "s0=" << sc[0] << endl;
   auto tmp1c = xgemm("N","N",Uc,diagonal_cmatrix(sc));
   auto tmp2c = xgemm("N","N",tmp1c,Vtc);
   auto diff_svdc = (cmat-tmp2c).normF();
   cmat.print("cmat");
   tmp2c.print("cmat_svd");
   cout << "diff_svdc=" << diff_svdc << endl;
   assert(diff_svdc < thresh);

   // --- linalg: EIG ---
   cout << "\ntest eig_solver" << endl;
   cout << "\nreal version:" << endl;
   mat = (mat + mat.H())*0.5;
   vector<double> e(n);
   matrix<double> v;
   eig_solver(mat, e, v);
   cout << "eig0=" << e[0] << endl;
   matrix<double> vt(v);
   auto diff = xgemm("T","N",vt,v) - identity_matrix<double>(n);
   v.print("V");
   diff.print("VtV-idn");
   cout << "normF=" << diff.normF() << endl;
   // Av-ve
   auto Av = xgemm("N","N",mat,v);
   auto ve = xgemm("N","N",v,diagonal_matrix(e));
   auto diff1 = Av - ve;
   diff1.print("Av-ve");
   double diff_eig = diff1.normF();
   cout << "normF=" << diff_eig << endl;
   assert(diff_eig < thresh);

   cout << "\ncomplex version:" << endl;
   cmat = (cmat + cmat.H())*0.5;
   matrix<complex<double>> vc;
   eig_solver(cmat, e, vc);
   cout << "eig0=" << e[0] << endl;
   auto  vtc = vc;
   auto diffc = xgemm("C","N",vtc,vc) - identity_matrix<complex<double>>(n);
   vc.print("Vc");
   diffc.print("VtV-idn");
   cout << "normF=" << diffc.normF() << endl;
   // Av-ve
   auto Avc = xgemm("N","N",cmat,vc);
   auto vce = xgemm("N","N",vc,diagonal_cmatrix(e));
   auto diff1c = Avc - vce;
   diff1c.print("Av-ve");
   double diff_eigc = diff1c.normF();
   cout << "normF=" << diff_eigc << endl;
   assert(diff_eigc < thresh);

 /**
   cout << "\n dgesdd_ test:" << endl;
   MKL_INT mm = 3000;
   MKL_INT nn = 3000;
   char jobz = 'S';
   MKL_INT minmn = std::min(mm,nn);
   MKL_INT info;

   double work;

   matrix<double> mat_A(mm,nn);
   matrix<double> mat_B(mm,nn);
   matrix<double> mat_U(mm,minmn);
   matrix<double> mat_VT(minmn,nn);
   matrix<MKL_INT> iwork(8*minmn,1);
   matrix<double> d(minmn,1);
   double workopt;
   MKL_INT lwork=-1;


   vector<double> s;
   matrix<double> U, Vt;
   
   const int ITER = 2;

   auto t0g = tools::get_time();
   for(int iii=0;iii<ITER;iii++)
   {
   auto mat_C= xgemm("N","N",mat_A,mat_B);
 
   }
   auto t1g = tools::get_time();
   double duration_gemm = tools::get_duration(t1g-t0g)/(ITER+0.0);
   std::cout<<"duration_gemm cost is: "<<duration_gemm <<" Second !"<<std::endl;


   ///////////////////////////////////////////////////////////////////////////////
   auto t0 = tools::get_time();
   for(int iii=0;iii<ITER;iii++)
   {
   //dgesdd_(&jobz, &mm, &nn, mat_A.data(), &mm, d.data(), mat_U.data(), &mm, mat_VT.data(), &minmn, &workopt, &lwork, iwork.data(), &info);
   //svd_solver(mat_A, s, U, Vt, 13);
   svd_solver(mat_A, s, U, Vt, 3);
 
   }
   auto t1 = tools::get_time();

   double duration_dgesdd = tools::get_duration(t1-t0)/(ITER+0.0);
   if(info !=0)
   {
       std::cout<<"dgesdd_ error"<<std::endl;
       exit(0);
   }

   std::cout<<"duration_dgesdd cost is: "<<duration_dgesdd <<" Second !"<<std::endl;
   ///////////////////////////////////////////////////////////////////////////////
   vector<double> e(nn);
   matrix<double> v;
   auto t0e = tools::get_time();
   for(int iii=0;iii<ITER;iii++)
   {
   //dgesdd_(&jobz, &mm, &nn, mat_A.data(), &mm, d.data(), mat_U.data(), &mm, mat_VT.data(), &minmn, &workopt, &lwork, iwork.data(), &info);
   //svd_solver(mat_A, s, U, Vt, 13);
   eig_solver(mat_A, e, v);
 
   }
   auto t1e = tools::get_time();

   double duration_e = tools::get_duration(t1e-t0e)/(ITER+0.0);

   std::cout<<"duration_e cost is: "<<duration_e <<" Second !"<<std::endl;
   ///////////////////////////////////////////////////////////////////////////////

   **/

   return 0;
}
