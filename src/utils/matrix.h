#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <cassert>
#include <random>
#include <iomanip>
#include <string>

namespace linalg{

class matrix{
   public:
      // constructors
      matrix(const int m, const int n): _rows(m), _cols(n){
	 _size = m*n;     
         _data = new double[_size];
	 std::fill_n(_data, _size, 0.0);
      }
      // desctructors
      ~matrix(){ delete[] _data; }
      // copy constructor
      matrix(const matrix& mat){
         _rows = mat._rows;
	 _cols = mat._cols;
	 _size = mat._size;
         _data = new double[_size];
	 std::copy_n(mat._data, _size, _data);
      }
      // copy assignment 
      matrix& operator =(const matrix& mat){
         if(this != &mat){
            _rows = mat._rows;
 	    _cols = mat._cols;	    
	    _size = mat._size;
	    delete[] _data;
	    _data = new double[_size];
	    std::copy_n(mat._data, _size, _data);
	 }
	 return *this;
      }
      // move constructor
      matrix(matrix&& mat){
         _rows = mat._rows;
	 _cols = mat._cols;
	 _size = mat._size;
	 _data = mat._data;
	 mat._rows = 0;
	 mat._cols = 0;
	 mat._size = 0;
	 mat._data = nullptr;
      }
      // move assignment 
      matrix& operator =(matrix&& mat){
         if(this != &mat){
            _rows = mat._rows;
 	    _cols = mat._cols;	    
	    _size = mat._size;
	    delete[] _data;
	    _data = mat._data;
	    mat._rows = 0;
	    mat._cols = 0;
	    mat._size = 0;
	    mat._data = nullptr;
	 }
	 return *this;
      }
      // access: A[i,j] - row major
      const double operator()(const int i, const int j) const{
	 assert(i>=0 && i<_rows && j>=0 && j<_cols);
	 return _data[i*_cols+j];
      } 
      double& operator()(const int i, const int j){
	 assert(i>=0 && i<_rows && j>=0 && j<_cols);
	 return _data[i*_cols+j];
      }
      // print
      void print(std::string name=""){
         std::cout << "matrix: " << name 
		   << " size=(" << _rows << "," << _cols << ")" 
		   << std::endl;
	 std::cout << std::scientific << std::setprecision(4); 
	 for(int i=0; i<_rows; i++){
   	    for(int j=0; j<_cols; j++){
	       std::cout << std::setw(12) << _data[i*_cols+j] << " ";
	    } 
	    std::cout << std::endl;
	 }
	 std::cout << std::defaultfloat;
      }
      // helpers
      inline int rows() const{ return _rows; }
      inline int cols() const{ return _cols; }
      inline int size() const{ return _size; }
      inline double* data() const{ return _data; }
      // basic mathematics of matrices
      std::vector<double> diagonal(){
	 assert(_rows == _cols);
         std::vector<double> diag(_rows);
	 for(int i=0; i<_rows; i++){
	    diag[i] = _data[i*_rows+i];
	 }
	 return diag;
      }
      double trace(){
	 assert(_rows == _cols);
         double tr = 0.0;
	 for(int i=0; i<_rows; i++){
	    tr += _data[i*_rows+i];
	 }
	 return tr;
      }
      // *,+,- operations
      matrix& operator *=(const double fac){
         std::transform(_data, _data+_size, _data,
			[fac](const double& x){return fac*x;});
	 return *this;
      }
      matrix& operator +=(const matrix& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const double& x, const double& y){return x+y;});
	 return *this;
      }
      matrix& operator -=(const matrix& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const double& x, const double& y){return x-y;});
	 return *this;
      }
      // friend
      friend matrix operator *(const double fac, const matrix& mat1);
      friend matrix operator *(const matrix& mat1, const double fac);
      friend matrix operator +(const matrix& mat1, const matrix& mat2);
      friend matrix operator -(const matrix& mat1, const matrix& mat2);
   private:
      int _rows, _cols, _size;
      double* _data;
};

// special matrices
matrix identity_matrix(const int n);

matrix diagonal_matrix(const std::vector<double>& diag);

//extern std::random_device rd;
extern std::seed_seq seeds;
extern std::default_random_engine generator;
matrix random_matrix(const int m, const int n);

} // linalg

#endif
