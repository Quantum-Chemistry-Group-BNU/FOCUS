#include <iostream>
#include "../settings/global.h"
#include "../utils/tools.h"
#include "../utils/matrix.h"
#include "../utils/linalg.h"
#include "tests.h"

using namespace std;
using namespace linalg;

int tests::test_tools(){
   cout << global::line_separator << endl;	
   cout << "test_tools" << endl;
   cout << global::line_separator << endl;
  
   const int nmax = 5;

   cout << "\ncanonical_pair0" << endl;
   for(int i=0; i<nmax; i++){
      for(int j=0; j<nmax; j++){
         int ii,jj,ij;
	 ij = tools::canonical_pair0(i,j);
	 tools::inverse_pair0(ij,ii,jj);
	 cout << "(" << i << "," << j << ")=>" << ij
	      << "=>" << ii << "," << jj << endl;	 
      }
   }

   cout << "\ncanonical_pair" << endl;
   for(int i=0; i<nmax; i++){
      for(int j=0; j<nmax; j++){
         int ii,jj,ij;
	 ij = tools::canonical_pair(i,j);
	 tools::inverse_pair(ij,ii,jj);
	 cout << "(" << i << "," << j << ")=>" << ij
	      << "=>" << ii << "," << jj << endl;	 
      }
   }

   // matrix
   matrix mat(3,3);
   mat(1,2) = 1.0;
   mat(2,0) = 3.0;
   cout << "mat\n" << mat << endl;
   matrix mat1(mat);
   cout << "mat1\n" << mat1 << endl;

   // random & identity;
   matrix iden = identity_matrix(4);
   matrix rd = random_matrix(4,3);
   cout << iden << endl;
   cout << "rd\n" << rd << endl;
   matrix rd2 = random_matrix(4,3);
   cout << "rd2\n" << rd2 << endl;

   dgemm("T","N",1.0,iden,rd,1.0,rd2);
   cout << "rd2\n" << rd2 << endl;
   dgemm("N","t",1.0,rd,rd2,1.0,iden);
   cout << "id\n" << iden << endl;
   // math
   iden *= 2;
   cout << "2*id\n" << iden << endl;
   iden += iden;
   cout << "+*id\n" << iden << endl;
   iden -= iden;
   cout << "-*id\n" << iden << endl;
   auto tmp = rd + rd2;
   cout << "rd+rd2\n" << tmp << endl;
   auto tmp1 = rd - rd2;
   cout << "rd-rd2\n" << tmp1 << endl;
   auto tmp2 = 3*rd;
   cout << "3*rd\n" << tmp2 << endl;
   auto tmp3 = 3*rd + rd2;
   cout << "3*rd+rd2\n" << tmp3 << endl;
   exit(1);

/*
   const int n = 10;
   matrix rd3 = random_matrix(n,n);
   matrix rd4(rd3);
   matrix rd5(n,n);
   dgemm("T","N",1.0,rd3,rd4,0.0,rd5);
   matrix v(rd5);
   vector<double> e(v.rows());
   eig(v,e);
   cout << "eig0=" << e[0] << endl;
   matrix vt(v);
   matrix idn(v.rows(),v.cols());
   dgemm("T","N",1.0,vt,v,0.0,idn);
   cout << "idn=" << idn << endl;   
   idn -= identity_matrix(n);
   cout << "Fnorm=" << idn.Fnorm() << endl;
   exit(1); 

   // Ax-xb
   matrix Av(n,n);
   dgemm("N","N",rd5,v,0.0,Av);
   matrix ve(n,n);
   dgemm("N","N",v,diagonal_matrix(e),0.0,ve);
   auto diff = Av - ve;
   cout << "Fnorm=" << diff.Fnorm() << endl; 
   exit(1);
*/
}
