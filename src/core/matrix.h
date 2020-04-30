#ifndef MATRIX_H
#define MATRIX_H

#include "serialization.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <string>
#include <vector>

namespace linalg{

// column-major matrix
class matrix{
   private:
      friend class boost::serialization::access;	   
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 int rows = _rows, cols = _cols, size = _size;
	 ar & rows;
	 ar & cols;
	 ar & size;
	 if(size != _size){
	    _rows = rows;
	    _cols = cols;
	    _size = size;
	    _data = new double[_size];
	 }
         for(int i=0; i<_size; i++){
	    ar & _data[i];
	 }
      }

   public:
      // constructors
      matrix(): _rows(0), _cols(0), _size(0), _data(nullptr) {};
      void resize(const int m, const int n){
	 _rows = m;
	 _cols = n;
	 _size = m*n;
	 delete[] _data;
         _data = new double[_size];
	 std::fill_n(_data, _size, 0.0);
      }
      matrix(const int m, const int n): _rows(m), _cols(n){
	 _size = m*n;     
         _data = new double[_size];
	 std::fill_n(_data, _size, 0.0);
      }
      // special constructor: copied from raw data
      matrix(const int m, const int n, const double* data): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new double[_size];
	 std::copy(data, data+_size, _data);
      }
      matrix(const int m, const int n, const double cval): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new double[_size];
	 std::fill(_data, _data+_size, cval);
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
	 return _data[j*_rows+i];
      } 
      double& operator()(const int i, const int j){
	 assert(i>=0 && i<_rows && j>=0 && j<_cols);
	 return _data[j*_rows+i];
      }
      // return the memory address 
      const double* addr(const int i, const int j) const{ // for f(const matrix& v)
         assert(i>=0 && i<_rows && j>=0 && j<_cols);
         return &_data[j*_rows+i];
      }
      double* addr(const int i, const int j){
         assert(i>=0 && i<_rows && j>=0 && j<_cols);
         return &_data[j*_rows+i];
      }
      // print
      void print(std::string name="") const{
         std::cout << "matrix: " << name 
		   << " size=(" << _rows << "," << _cols << ")" 
		   << std::endl;
	 std::cout << std::scientific << std::setprecision(4); 
	 for(int i=0; i<_rows; i++){
   	    for(int j=0; j<_cols; j++){
	       std::cout << std::setw(12) << _data[j*_rows+i] << " ";
	    } 
	    std::cout << std::endl;
	 }
	 std::cout << std::defaultfloat;
      }
      // save
      void save_text(const std::string& fname, const int prec=4) const;
      // binary
      void save(const std::string& fname) const; 
      void load(const std::string& fname); 
      // helpers
      inline int rows() const{ return _rows; }
      inline int cols() const{ return _cols; }
      inline int size() const{ return _size; }
      inline double* data() const{ return _data; }
      // basic mathematics of matrices
      std::vector<double> diagonal() const{
	 assert(_rows == _cols);
         std::vector<double> diag(_rows);
	 for(int i=0; i<_rows; i++){
	    diag[i] = _data[i*_rows+i];
	 }
	 return diag;
      }
      double trace() const{
	 assert(_rows == _cols);
         double tr = 0.0;
	 for(int i=0; i<_rows; i++){
	    tr += _data[i*_rows+i];
	 }
	 return tr;
      }
      // transpose
      matrix transpose() const{
         matrix At(_cols,_rows);
         for(int j=0; j<_rows; j++){
            for(int i=0; i<_cols; i++){
      	       At(i,j) = _data[i*_rows+j];
            }
         }
         return At;
      }
      // row operations
      const double* col(const int i) const{
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      double* col(const int i){
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      void col_scale(const int icol, const double fac){
         std::transform(&_data[icol*_rows], (&_data[icol*_rows])+_cols, 
			&_data[icol*_rows],
			[fac](const double& x){return fac*x;});
      }
      // =,*,+,- operations
      matrix& operator =(const double val){
	 std::fill_n(_data, _size, val);
	 return *this;
      }
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
      matrix operator -() const{
	 matrix mat(_rows,_cols);
         std::transform(_data, _data+_size, mat._data,
			[](const double& x){return -x;});
	 return mat;
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
matrix zero_matrix(const int m, const int n);

matrix identity_matrix(const int n);

matrix diagonal_matrix(const std::vector<double>& diag);

matrix random_matrix(const int m, const int n);

} // linalg

#endif
