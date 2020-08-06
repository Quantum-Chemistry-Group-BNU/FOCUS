#include "../core/tools.h"
#include "../core/matrix.h"
#include "tests.h"
#include <complex>

using namespace linalg;
using namespace std;

int tests::test_matrix(){
   cout << endl;
   cout << tools::line_separator << endl;	
   cout << "tests::test_matrix" << endl;
   cout << tools::line_separator << endl;
  
   // --- real matrix --- 
   matrix<double> mat(3,3);
   mat(1,2) = 1.0;
   mat(2,0) = 3.0;
   mat.print("mat");
   
   matrix<double> mat1(mat);
   mat1.print("mat1");
   
   // random & identity;
   matrix<double> iden = identity_matrix<double>(4);
   matrix<double> rd = random_matrix(4,3);
   iden.print("iden");
   rd.print("rd");
   matrix<double> rd2 = random_matrix(4,3);
   rd2.print("rd2");

   // simple math for real matrix
   iden *= 2;
   iden.print("2*id");
   iden += iden;
   iden.print("+*id");
   iden -= iden;
   iden.print("-*id");
   auto tmp = rd + rd2;
   tmp.print("rd+rd2");
   rd.print("rd");
   rd2.print("rd2");
   auto tmp1 = rd - rd2;
   tmp1.print("rd-rd2");
   auto tmp2 = 3*rd;
   tmp2.print("3*rd");
   auto tmp3 = 3*rd + rd2;
   tmp3.print("3*rd+rd2");

   // --- complex matrix --- 
   const complex<double> i(0.0,1.0);
   matrix<complex<double>> cmat(3,3);
   cmat(1,2) = 1.0;
   cmat(2,0) = 3.0;
   cmat(2,1) = 1.0+1.0*i;
   cmat.print("cmat");
   
   matrix<complex<double>> cmat1(cmat);
   cmat1.print("cmat1");

   // simple +/-/*
   auto cmat2 = i*cmat;
   cmat2.print("cmat2");

   auto cmat3 = i*mat;
   cmat3.print("cmat3");

   // simple math for real matrix
   cmat.print("cmat");
   auto cmatx(cmat);
   cmatx *= 2;
   cmatx.print("2*cmat");
   cmatx += cmat;
   cmatx.print("+*cmat");
   cmatx -= cmat;
   cmatx.print("-*cmat");
  
   cout << endl;
   cmat.print("cmat"); 
   cmat2.print("cmat2"); 
   auto tmp0c = cmat + cmat2;
   tmp0c.print("cmat+cmat2");
   auto tmp1c = cmat - cmat2;
   tmp1c.print("cmat-cmat2");
   auto tmp2c = 3*cmat;
   tmp2c.print("3*cmat");
   auto tmp3c = 3*cmat + cmat2;
   tmp3c.print("3*cmat+cmat2");
   cout << endl;

   // conjugation
   iden(1,0) = 2;
   iden.print("iden");
   iden.conj().print("iden.conj");
   iden.T().print("iden.T");
   iden.H().print("iden.H");

   cmat.print("cmat");
   cmat.conj().print("cmat.conj");
   cmat.T().print("cmat.T");
   cmat.H().print("cmat.H");
   cmat.real().print("cmat.real");
   cmat.imag().print("cmat.imag");
   cout << endl;

   // io
   iden.save_text("iden");
   cmat.save_text("cmat");
   
   iden.save("iden");
   cmat.save("cmat");
   iden(1,1) = 1;
   iden.print("iden_modified");
   iden.load("iden");
   iden.print("iden_load");
   
   matrix<double> empty;
   empty.load("iden");
   empty.print("empty");
   
   cmat(1,0) = {1,0};
   cmat.load("cmat");
   cmat.print("cmat_load");
   
   // from diagonal
   vector<double> v({-1,1,2});
   auto w = diagonal_matrix(v);
   w.print("w");
   
   vector<complex<double>> vc({-1,{1,1},2});
   auto wc = diagonal_matrix(vc);
   wc.print("wc");
   
   return 0;
}
