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
   mat.print("mat");
   matrix mat1(mat);
   mat.print("mat1");

   // random & identity;
   matrix iden = identity_matrix(4);
   matrix rd = random_matrix(4,3);
   iden.print();
   rd.print("rd");
   matrix rd2 = random_matrix(4,3);
   rd2.print("rd2");

   // dgemm
   dgemm("T","N",1.0,iden,rd,1.0,rd2);
   rd2.print("rd2");
   dgemm("N","t",1.0,rd,rd2,1.0,iden);
   iden.print("id");
   
   // math
   iden *= 2;
   iden.print("2*id");
   iden += iden;
   iden.print("+*id");
   iden -= iden;
   iden.print("-*id");
   auto tmp = rd + rd2;
   iden.print("rd+rd2");
   auto tmp1 = rd - rd2;
   tmp1.print("rd-rd2");
   auto tmp2 = 3*rd;
   tmp2.print("3*rd");
   auto tmp3 = 3*rd + rd2;
   tmp3.print("3*rd+rd2");

   // linalg: Av=ve
   const int n = 6;
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
   idn -= identity_matrix(n);
   idn.print("idn");
   cout << "normF=" << normF(idn) << endl;

   // Ax-xb
   matrix Av(n,n);
   dgemm("N","N",1.0,rd5,v,0.0,Av);
   matrix ve(n,n);
   dgemm("N","N",1.0,v,diagonal_matrix(e),0.0,ve);
   auto diff = Av - ve;
   diff.print("diff");
   cout << "normF=" << normF(diff) << endl; 

   return 0;
}
