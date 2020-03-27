#include <random>
#include <fstream>
#include "matrix.h"

using namespace std;
using namespace linalg;

matrix linalg::operator *(const double fac, const matrix& mat1){
   matrix mat(mat1.rows(),mat1.cols());
   transform(mat1._data, mat1._data+mat._size, mat._data,
	     [fac](const double& x){return fac*x;});
   return mat;
}

matrix linalg::operator *(const matrix& mat1, const double fac){
   return fac*mat1;
}

matrix linalg::operator +(const matrix& mat1, const matrix& mat2){
   assert(mat1._size == mat2._size);
   matrix mat(mat1.rows(),mat1.cols());
   std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
		  [](const double& x, const double& y){return x+y;});
   return mat;
}

matrix linalg::operator -(const matrix& mat1, const matrix& mat2){
   assert(mat1._size == mat2._size);
   matrix mat(mat1.rows(),mat1.cols());
   std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
		  [](const double& x, const double& y){return x-y;});
   return mat;
}

matrix linalg::zero_matrix(const int m, const int n){
   matrix zero(m,n);
   return zero;
}

matrix linalg::identity_matrix(const int n){
   matrix iden(n,n);
   for(int i=0; i<n; i++)
      iden(i,i) = 1.0;
   return iden;
}

matrix linalg::diagonal_matrix(const vector<double>& diag){
   int n = diag.size();
   matrix mat(n,n);
   for(int i=0; i<n; i++) 
      mat(i,i) = diag[i];	   
   return mat;
}

//std::random_device linalg::rd; // non-deterministic hardware gen
std::seed_seq linalg::seeds{0}; //linalg::rd()};
std::default_random_engine linalg::generator(linalg::seeds);

matrix linalg::random_matrix(const int m, const int n){
   std::uniform_real_distribution<double> dist(-1,1);
   matrix rand(m,n);
   // column major
   for(int j=0; j<n; j++){
      for(int i=0; i<m; i++){
         rand(i,j) = dist(linalg::generator);
      }
   }
   return rand;
}

void matrix::save_text(const string& fname) const{
   ofstream file(fname+".txt"); 
   file << defaultfloat << setprecision(4); 
   for(int i=0; i<_rows; i++){
      for(int j=0; j<_cols; j++){
         file << _data[j*_rows+i] << " ";
      } 
      file << endl;
   }
   file.close();
}
