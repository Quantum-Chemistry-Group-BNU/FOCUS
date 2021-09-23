#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <algorithm>
#include <cassert>
#include <iomanip>
#include <string>
#include <vector>
#include <complex>
#include <fstream>
#include "tools.h"
#include "serialization.h"

namespace linalg{

template <typename Tm>
struct BaseMatrix{
   public:
      virtual int rows() const = 0;
      virtual int cols() const = 0;
      virtual size_t size() const = 0;
      virtual const Tm* data() const = 0;
      virtual Tm* data() = 0;
      virtual const Tm operator()(const int i, const int j) const = 0;
      virtual Tm& operator()(const int i, const int j) = 0;
};

// column-major matrix
template <typename Tm>
struct matrix : public BaseMatrix<Tm> {
   private:
      friend class boost::serialization::access;	   
      template <class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 int rows = _rows, cols = _cols, size = _size;
	 ar & rows;
	 ar & cols;
	 ar & size;
	 if(size != _size){
	    _rows = rows;
	    _cols = cols;
	    _size = size;
	    delete[] _data;
	    _data = new Tm[_size];
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
      // some special constructors: 
      // 1. copied from raw data
      matrix(const int m, const int n, const Tm* pdata): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new Tm[_size];
	 std::copy(pdata, pdata+_size, _data);
      }
      // 2. from constant value
      matrix(const int m, const int n, const Tm cval): _rows(m), _cols(n){
	 _size = m*n;
 	 _data = new Tm[_size];
	 std::fill(_data, _data+_size, cval);
      }
      // 3. from vector of vector
      matrix(const std::vector<std::vector<Tm>>& vs){
         _cols = vs.size();
	 _rows = vs[0].size();
	 _size = _rows*_cols;
	 _data = new Tm[_size];
	 for(int j=0; j<_cols; j++){
	    std::copy(vs[j].cbegin(), vs[j].cend(), &_data[j*_rows]);
	 }
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
      void save_text(const std::string& fname, const int prec=4) const{
	 std::ofstream file(fname+".txt"); 
         file << std::defaultfloat << std::setprecision(prec); 
         for(int i=0; i<_rows; i++){
            for(int j=0; j<_cols; j++){
               file << _data[j*_rows+i] << " ";
            } 
            file << std::endl;
         }
         file.close();
      }
      // binary
      void save(const std::string& fname) const{
	 std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive saver(ofs);
         saver << *this;
      } 
      void load(const std::string& fname){
	 std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive loader(ifs);
         loader >> *this;
      }
      // helpers
      int rows() const{ return _rows; }
      int cols() const{ return _cols; }
      size_t size() const{ return _size; }
      const Tm* data() const{ return _data; }
      Tm* data(){ return _data; }
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
      // complex case: conjugate & Hermitian conjugate
      matrix conj() const{
	 matrix<Tm> Ac(_rows,_cols);
         std::transform(_data, _data+_size, Ac._data,
			[](const Tm& x){ return tools::conjugate(x); });
	 return Ac;
      }
      matrix H() const{
         matrix<Tm> Ah(_cols,_rows);
         for(int j=0; j<_rows; j++){
            for(int i=0; i<_cols; i++){
      	       Ah(i,j) = tools::conjugate(_data[i*_rows+j]);
            }
         }
         return Ah;
      }
      // extract real & imag parts
      matrix<double> real() const{
	 matrix<double> matr(_rows,_cols);
	 std::transform(_data, _data+_size, matr._data,
			[](const Tm& x){ return std::real(x); });
	 return matr;
      }
      matrix<double> imag() const{
	 matrix<double> mati(_rows,_cols);
	 std::transform(_data, _data+_size, mati._data,
			[](const Tm& x){ return std::imag(x); });
	 return mati;
      }
      // convert real to complex
      matrix<std::complex<double>> as_complex() const{
	 matrix<std::complex<double>> mat(_rows,_cols);
	 std::transform(_data, _data+_size, mat._data,
			[](const Tm& x){ return x; });
	 return mat;
      }
      // col operations
      const Tm* col(const int i) const{
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      Tm* col(const int i){
	 assert(i>=0 && i<_cols);
	 return &_data[i*_rows];
      }
      // scale (used in blockMatrix)
      void rowscale(const std::vector<double>& phases){
	 if(_cols == 0) return;
	 assert(phases.size() == _rows);
         for(int ic=0; ic<_cols; ic++){
	    std::transform(this->col(ic), this->col(ic)+_rows, phases.begin(), this->col(ic),
			   [](const Tm& x, const double& y){ return x*y; });
	 }
      }
      void colscale(const std::vector<double>& phases){
         if(_rows == 0) return;
	 assert(phases.size() == _cols);
         for(int ic=0; ic<_cols; ic++){
	    double phase = phases[ic];
	    std::transform(this->col(ic), this->col(ic)+_rows, this->col(ic),
			   [phase](const Tm& x){ return x*phase; });
	 }
      }
      // ZL@20210421: reorder row/col/rowcol 
      // iop=0: i-th pos in mat[new] is the idx[i]-th pos in mat[old]
      matrix reorder_row(const std::vector<int>& idx, const bool iop=0){
	 matrix<Tm> rmat(_rows,_cols);
	 if(iop == 0){
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(i,j) = (*this)(idx[i],j);
               }
            }
	 }else{
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(idx[i],j) = (*this)(i,j);
               }
            }
	 }
         return rmat;
      }
      matrix reorder_col(const std::vector<int>& idx, const bool iop=0){
	 matrix<Tm> rmat(_rows,_cols);
	 if(iop == 0){
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(i,j) = (*this)(i,idx[j]);
               }
            }
	 }else{
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(i,idx[j]) = (*this)(i,j);
               }
            }
	 }
         return rmat;
      }
      matrix reorder_rowcol(const std::vector<int>& idx1,
		      	    const std::vector<int>& idx2,
			    const bool iop=0){
	 matrix<Tm> rmat(_rows,_cols);
	 if(iop == 0){
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(i,j) = (*this)(idx1[i],idx2[j]);
               }
            }
	 }else{
            for(int j=0; j<_cols; j++){
               for(int i=0; i<_rows; i++){
      	          rmat(idx1[i],idx2[j]) = (*this)(i,j);
               }
            }
	 }
         return rmat;
      }
      // =,*,+,- operations
      matrix& operator =(const Tm cval){
	 std::fill_n(_data, _size, cval);
	 return *this;
      }
      matrix& operator *=(const Tm fac){
         std::transform(_data, _data+_size, _data,
			[fac](const Tm& x){ return fac*x; });
	 return *this;
      }
      matrix& operator +=(const matrix<Tm>& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const Tm& x, const Tm& y){ return x+y; });
	 return *this;
      }
      matrix& operator -=(const matrix<Tm>& mat){
         std::transform(_data, _data+_size, mat._data, _data,
			[](const Tm& x, const Tm& y){ return x-y; });
	 return *this;
      }
      matrix operator -() const{
	 matrix<Tm> mat(_rows,_cols);
         std::transform(_data, _data+_size, mat._data,
			[](const Tm& x){ return -x; });
	 return mat;
      }
      // simple */+/-
      friend matrix operator *(const Tm fac, const matrix<Tm>& mat1){
         matrix<Tm> mat(mat1.rows(),mat1.cols());
	 std::transform(mat1._data, mat1._data+mat._size, mat._data,
      	                [fac](const Tm& x){ return fac*x; });
         return mat;
      }
      friend matrix operator *(const matrix<Tm>& mat1, const Tm fac){
         return fac*mat1;
      }
      friend matrix operator +(const matrix<Tm>& mat1, const matrix<Tm>& mat2){
         assert(mat1._size == mat2._size);
         matrix<Tm> mat(mat1.rows(),mat1.cols());
         std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
      		        [](const Tm& x, const Tm& y){ return x+y; });
         return mat;
      }
      friend matrix operator -(const matrix<Tm>& mat1, const matrix<Tm>& mat2){
         assert(mat1._size == mat2._size);
         matrix<Tm> mat(mat1.rows(),mat1.cols());
         std::transform(mat1._data, mat1._data+mat1._size, mat2._data, mat._data,
              	        [](const Tm& x, const Tm& y){ return x-y; });
         return mat;
      }
      // M(l,r) = M1(bar{l},bar{r})^* given parity of qr and qc
      matrix time_reversal(const int pr, const int pc) const{
         matrix<Tm> blk(_rows,_cols);
         // even-even block:
         //    <e|\bar{O}|e> = p{O} <e|O|e>^*
         if(pr == 0 && pc == 0){
            blk = this->conj();
         // even-odd block:
         //    [A,B] -> p{O}[B*,-A*]  
         // tA = <e|\bar{O}|o> = p{O} <e|O|\bar{o}>^*
         // tB = <e|\bar{O}|\bar{o}> = p{O} <e|O|o>^* (-1)
         }else if(pr == 0 && pc == 1){
            assert(_cols%2 == 0);
            int dc2 = _cols/2;
            // copy blocks <e|O|o>^*
            for(int ic=0; ic<dc2; ic++){
               std::transform(this->col(ic),this->col(ic)+_rows,blk.col(ic+dc2),
                	      [](const Tm& x){ return -tools::conjugate(x); });
            }
            // copy blocks <e|O|\bar{o}>
            for(int ic=0; ic<dc2; ic++){
               std::transform(this->col(ic+dc2),this->col(ic+dc2)+_rows,blk.col(ic),
               	              [](const Tm& x){ return tools::conjugate(x); });
            }
         // odd-even block:
         //    [A]        [ B*]
         //    [ ] -> p{O}[   ]
         //    [B]        [-A*] 
         }else if(pr == 1 && pc == 0){
            assert(_rows%2 == 0);
            int dr2 = _rows/2;
            for(int ic=0; ic<_cols; ic++){
               std::transform(this->col(ic),this->col(ic)+dr2,blk.col(ic)+dr2,
               	              [](const Tm& x){ return -tools::conjugate(x); });
               std::transform(this->col(ic)+dr2,this->col(ic)+_rows,blk.col(ic),
               	              [](const Tm& x){ return tools::conjugate(x); });
            }
         // odd-odd block:
         //    [A B]        [ D* -C*]
         //    [   ] -> p{O}[       ]
         //    [C D]        [-B*  A*]
         }else if(pr == 1 && pc == 1){
            assert(_rows%2 == 0 && _cols%2 == 0);
            int dr2 = _rows/2, dc2 = _cols/2;
            for(int ic=0; ic<dc2; ic++){
               std::transform(this->col(ic),this->col(ic)+dr2,blk.col(ic+dc2)+dr2,
               	              [](const Tm& x){ return tools::conjugate(x); });
               std::transform(this->col(ic)+dr2,this->col(ic)+_rows,blk.col(ic+dc2),
               	              [](const Tm& x){ return -tools::conjugate(x); });
            }
            for(int ic=0; ic<dc2; ic++){
               std::transform(this->col(ic+dc2),this->col(ic+dc2)+dr2,blk.col(ic)+dr2,
               	              [](const Tm& x){ return -tools::conjugate(x); });
               std::transform(this->col(ic+dc2)+dr2,this->col(ic+dc2)+_rows,blk.col(ic),
               	              [](const Tm& x){ return tools::conjugate(x); });
            }
         } // (pr,pc)
         return blk;
      }
   public:
      int _rows, _cols;
      size_t _size;
      Tm* _data;
};

// operator */+/- for conversion between real & complex matrices
matrix<std::complex<double>> operator *(const std::complex<double> fac, 
				        const matrix<double>& mat1);
matrix<std::complex<double>> operator *(const matrix<double>& mat1,
					const std::complex<double> fac);
matrix<std::complex<double>> operator +(const matrix<double>& mat1,
					const matrix<std::complex<double>>& mat2);
matrix<std::complex<double>> operator +(const matrix<std::complex<double>>& mat1,
					const matrix<double>& mat2);
matrix<std::complex<double>> operator -(const matrix<double>& mat1,
					const matrix<std::complex<double>>& mat2);
matrix<std::complex<double>> operator -(const matrix<std::complex<double>>& mat1,
					const matrix<double>& mat2);

// special matrices
template <typename Tm>
matrix<Tm> diagonal_matrix(const std::vector<Tm>& diag){
   int n = diag.size();
   matrix<Tm> mat(n,n);
   for(int i=0; i<n; i++) 
      mat(i,i) = diag[i];	   
   return mat;
}
// convert diagonal double to complex 
matrix<std::complex<double>> diagonal_cmatrix(const std::vector<double>& diag);

template <typename Tm>
matrix<Tm> identity_matrix(const int n){
   matrix<Tm> iden(n,n);
   for(int i=0; i<n; i++)
      iden(i,i) = 1.0;
   return iden;
}

// random matrix
template <typename Tm>
matrix<Tm> random_matrix(const int m, const int n){};

template <>
inline matrix<double> random_matrix(const int m, const int n){
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

template <>
inline matrix<std::complex<double>> random_matrix(const int m, const int n){
   const std::complex<double> i(0.0,1.0);
   matrix<std::complex<double>> rand(m,n);
   rand = random_matrix<double>(m,n) + i*random_matrix<double>(m,n);
   return rand;
}

} // linalg

#endif
