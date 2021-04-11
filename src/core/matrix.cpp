#include "matrix.h"
#include "tools.h"

using namespace std;
using namespace linalg;

// operator */+/- for conversion between real & complex
matrix<complex<double>> linalg::operator *(const complex<double> fac,
				           const matrix<double>& mat1){
   matrix<complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat._size, mat._data,
     	     [fac](const double& x){return fac*x;});
   return mat;
}
matrix<complex<double>> linalg::operator *(const matrix<double>& mat1,
				           const complex<double> fac){
   return fac*mat1;
}

matrix<complex<double>> linalg::operator +(const matrix<double>& mat1,
				           const matrix<complex<double>>& mat2){
   assert(mat1._size == mat2._size);
   matrix<complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      	     [](const double& x, const complex<double>& y){return x+y;});
   return mat;
} 
matrix<complex<double>> linalg::operator +(const matrix<complex<double>>& mat1,
				   	   const matrix<double>& mat2){
   assert(mat1._size == mat2._size);
   matrix<complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      	     [](const complex<double>& x, const double& y){return x+y;});
   return mat;
}

matrix<complex<double>> linalg::operator -(const matrix<double>& mat1,
				   	   const matrix<complex<double>>& mat2){
   assert(mat1._size == mat2._size);
   matrix<complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      	     [](const double& x, const complex<double>& y){return x-y;});
   return mat;
} 
matrix<complex<double>> linalg::operator -(const matrix<complex<double>>& mat1,
				   	   const matrix<double>& mat2){
   assert(mat1._size == mat2._size);
   matrix<complex<double>> mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      	     [](const complex<double>& x, const double& y){return x-y;});
   return mat;
}

// special matrices
matrix<complex<double>> linalg::diagonal_cmatrix(const vector<double>& diag){
   int n = diag.size();
   matrix<complex<double>> mat(n,n);
   for(int i=0; i<n; i++) 
      mat(i,i) = diag[i];	   
   return mat;
}
