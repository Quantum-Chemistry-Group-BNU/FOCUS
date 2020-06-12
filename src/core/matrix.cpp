#include "matrix.h"
#include "tools.h"

using namespace std;
using namespace linalg;

// operator * for conversion between real & complex
matrix<std::complex<double>> linalg::operator *(const std::complex<double> fac,
				        	const matrix<double>& mat1){
   matrix<std::complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat._size, mat._data,
     	     [fac](const double& x){return fac*x;});
   return mat;
}

matrix<std::complex<double>> linalg::operator *(const matrix<double>& mat1,
				        	const std::complex<double> fac){
   return fac*mat1;
}

// special matrices
matrix<complex<double>> linalg::diagonal_cmatrix(const vector<double>& diag){
   int n = diag.size();
   matrix<complex<double>> mat(n,n);
   for(int i=0; i<n; i++) 
      mat(i,i) = diag[i];	   
   return mat;
}

matrix<double> linalg::identity_matrix(const int n){
   matrix<double> iden(n,n);
   for(int i=0; i<n; i++)
      iden(i,i) = 1.0;
   return iden;
}
matrix<complex<double>> linalg::identity_cmatrix(const int n){
   matrix<complex<double>> iden(n,n);
   for(int i=0; i<n; i++)
      iden(i,i) = 1.0;
   return iden;
}

matrix<double> linalg::random_matrix(const int m, const int n){
   std::uniform_real_distribution<double> dist(-1,1);
   matrix<double> rand(m,n);
   // column major
   for(int j=0; j<n; j++){
      for(int i=0; i<m; i++){
         rand(i,j) = dist(tools::generator);
      }
   }
   return rand;
}
