#ifndef MATRIX_H
#define MATRIX_H

#include "serialization.h"
#include <iostream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <string>
#include <vector>
#include <complex>

namespace linalg{

// simple strategy to enable both cases
inline double conjugate(const double x){ return x; }
inline std::complex<double> conjugate(const std::complex<double> x){ return conj(x); };

// column-major matrix
template <typename Tm>
struct matrix{
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
         _data = new Tm[_size];
	 std::fill_n(_data, _size, 0.0);
      }
      matrix(const int m, const int n): _rows(m), _cols(n){
	 _size = m*n;     
         _data = new Tm[_size];
	 std::fill_n(_data, _size, 0.0);
      }
      // special constructor: copied from raw data
      matrix(const int m, const int n, const Tm* data): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new Tm[_size];
	 std::copy(data, data+_size, _data);
      }
      matrix(const int m, const int n, const Tm cval): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new Tm[_size];
	 std::fill(_data, _data+_size, cval);
      }
      // desctructors
      ~matrix(){ delete[] _data; }
      // copy constructor
      matrix(const matrix& mat){
         _rows = mat._rows;
	 _cols = mat._cols;
	 _size = mat._size;
         _data = new Tm[_size];
	 std::copy_n(mat._data, _size, _data);
      }
      // copy assignment 
      matrix& operator =(const matrix& mat){
         if(this != &mat){
            _rows = mat._rows;
 	    _cols = mat._cols;	    
	    _size = mat._size;
	    delete[] _data;
	    _data = new Tm[_size];
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
      const Tm operator()(const int i, const int j) const{
	 assert(i>=0 && i<_rows && j>=0 && j<_cols);
	 return _data[j*_rows+i];
      } 
      Tm& operator()(const int i, const int j){
	 assert(i>=0 && i<_rows && j>=0 && j<_cols);
	 return _data[j*_rows+i];
      }
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << "matrix: " << name 
		   << " size=(" << _rows << "," << _cols << ")" 
		   << std::endl;
	 std::cout << std::scientific << std::setprecision(prec); 
	 for(int i=0; i<_rows; i++){
   	    for(int j=0; j<_cols; j++){
	       std::cout << std::setw(12+prec) << _data[j*_rows+i] << " ";
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
      inline Tm* data() const{ return _data; }
      // basic mathematics of matrices
      std::vector<Tm> diagonal() const{
	 assert(_rows == _cols);
         std::vector<Tm> diag(_rows);
	 for(int i=0; i<_rows; i++){
	    diag[i] = _data[i*_rows+i];
	 }
	 return diag;
      }
      Tm trace() const{
	 assert(_rows == _cols);
         Tm tr = 0.0;
	 for(int i=0; i<_rows; i++){
	    tr += _data[i*_rows+i];
	 }
	 return tr;
      }
      // transpose
      matrix T() const{
         matrix<Tm> At(_cols,_rows);
         for(int j=0; j<_rows; j++){
            for(int i=0; i<_cols; i++){
      	       At(i,j) = _data[i*_rows+j];
            }
         }
         return At;
      }
      // complex case
      matrix conj() const{
	 matrix<Tm> mat(_rows,_cols);
         std::transform(_data, _data+_size, mat._data,
			[](const Tm& x){return conjugate(x);});
	 return mat;
      }
      matrix H() const{
         matrix<Tm> Ah(_cols,_rows);
         for(int j=0; j<_rows; j++){
            for(int i=0; i<_cols; i++){
      	       Ah(i,j) = conjugate(_data[i*_rows+j]);
            }
         }
         return Ah;
      }
      // row operations
      const Tm* col(const int i) const{
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      Tm* col(const int i){
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      // =,*,+,- operations
      matrix& operator =(const Tm cval){
	 std::fill_n(_data, _size, cval);
	 return *this;
      }
      matrix& operator *=(const Tm fac){
         std::transform(_data, _data+_size, _data,
			[fac](const Tm& x){return fac*x;});
	 return *this;
      }
      matrix& operator +=(const matrix<Tm>& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const Tm& x, const Tm& y){return x+y;});
	 return *this;
      }
      matrix& operator -=(const matrix<Tm>& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const Tm& x, const Tm& y){return x-y;});
	 return *this;
      }
      matrix operator -() const{
	 matrix<Tm> mat(_rows,_cols);
         std::transform(_data, _data+_size, mat._data,
			[](const Tm& x){return -x;});
	 return mat;
      }
      // simple */+/-
      friend matrix operator *(const Tm fac, const matrix<Tm>& mat1){
         matrix<Tm> mat(mat1.rows(),mat1.cols());
         transform(mat1._data, mat1._data+mat._size, mat._data,
      	           [fac](const Tm& x){return fac*x;});
         return mat;
      }
      friend matrix operator *(const matrix<Tm>& mat1, const Tm fac){
         return fac*mat1;
      }
      friend matrix operator +(const matrix<Tm>& mat1, const matrix<Tm>& mat2){
         assert(mat1._size == mat2._size);
         matrix<Tm> mat(mat1.rows(),mat1.cols());
         std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      		        [](const Tm& x, const Tm& y){return x+y;});
         return mat;
      }
      friend matrix operator -(const matrix<Tm>& mat1, const matrix<Tm>& mat2){
         assert(mat1._size == mat2._size);
         matrix<Tm> mat(mat1.rows(),mat1.cols());
         std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
              	        [](const Tm& x, const Tm& y){return x-y;});
         return mat;
      }
   public:
      int _rows, _cols, _size;
      Tm* _data;
};

// operator * for conversion between real & complex
matrix<std::complex<double>> operator *(const std::complex<double> fac, 
				        const matrix<double>& mat1);

matrix<std::complex<double>> operator *(const matrix<double>& mat1,
					const std::complex<double> fac);

// special matrices
template <typename Tm>
matrix<Tm> diagonal_matrix(const std::vector<Tm>& diag);

matrix<double> identity_matrix(const int n);

matrix<double> random_matrix(const int m, const int n);

} // linalg

#endif
