#include <fstream>
#include "matrix.h"
#include "tools.h"
#include "serialization.h"

using namespace std;
using namespace linalg;

// io
template <typename Tm>
void matrix<Tm>::save_text(const string& fname, const int prec) const{
   ofstream file(fname+".txt"); 
   file << defaultfloat << setprecision(prec); 
   for(int i=0; i<_rows; i++){
      for(int j=0; j<_cols; j++){
         file << _data[j*_rows+i] << " ";
      } 
      file << endl;
   }
   file.close();
}

// binary
template <typename Tm>
void matrix<Tm>::save(const string& fname) const{
   ofstream ofs(fname, std::ios::binary);
   boost::archive::binary_oarchive save(ofs);
   save << *this;
}

template <typename Tm>
void matrix<Tm>::load(const string& fname){
   ifstream ifs(fname, std::ios::binary);
   boost::archive::binary_iarchive load(ifs);
   load >> *this;
}

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
template <typename Tm>
matrix<Tm> linalg::diagonal_matrix(const vector<Tm>& diag){
   int n = diag.size();
   matrix<Tm> mat(n,n);
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
