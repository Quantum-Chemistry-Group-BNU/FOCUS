#ifndef TENSOR_DENSE_H
#define TENSOR_DENSE_H

#include <cassert>
#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

//
// Dense tensor stored in FORTRAN ORDER to interface with BLAS/LAPACK
// They are just wrapper for data, so they are free to be copied/moved. 
//
	
// O[l,r]
template <typename Tm>   
struct dtensor2 : public linalg::BaseMatrix<Tm> {
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dim0 & dim1 & _off & _size;
      }
      int _addr(const int i0, const int i1) const{ 
	 return _off+i0+dim0*i1; 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor2(): dim0(0), dim1(0), _off(0), _size(0), _data(nullptr) {}
      void setup_dims(const int _dim0, const int _dim1, const size_t off){
	 dim0 = _dim0;
	 dim1 = _dim1; 
	 _off = off;
	 _size = dim0*dim1; 
      }
      void setup_data(Tm* data){ _data = data; }
      const Tm operator()(const int i0, const int i1) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 return _data[_addr(i0,i1)];
      }
      Tm& operator()(const int i0, const int i1){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 return _data[_addr(i0,i1)];
      } 
      size_t size() const{ return _size; };
      Tm* data() const{ return _data+_off; }
      // in-place operation
      void conjugate(){
         std::transform(_data+_off, _data+_off+_size, _data+_off,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor2: " << name 
		   << " size=(" << dim0 << "," << dim1 << ")" 
		   << std::endl;
	 std::cout << std::scientific << std::setprecision(prec); 
	 for(int i0=0; i0<dim0; i0++){
   	    for(int i1=0; i1<dim1; i1++){
	       std::cout << std::setw(12+prec) << (*this)(i0,i1) << " ";
	    } // i1
	    std::cout << std::endl;
	 } // i0
	 std::cout << std::defaultfloat;
      }
      linalg::matrix<Tm> to_matrix() const{
	 linalg::matrix<Tm> mat(dim0,dim1,_data+_off);
	 return mat;
      }
      // assignment
      dtensor2<Tm>& operator =(const linalg::matrix<Tm>& mat){
         assert(dim0 == mat.rows() && dim1 == mat.cols());
         std::copy_n(mat.data(), _size, _data+_off);
         return *this;
      }
      // not allowed
      dtensor2<Tm>& operator =(linalg::matrix<Tm>&& mat) = delete;
      dtensor2<Tm>& operator +=(const linalg::matrix<Tm>& mat){
         assert(dim0 == mat.rows() && dim1 == mat.cols());
         std::transform(_data+_off, _data+_off+_size, mat.data(), _data+_off,
			[](const Tm& x, const Tm& y){ return x+y; });
         return *this;
      }
      // interface with xgemm
      int rows() const{ return dim0; }
      int cols() const{ return dim1; }
   public:
      int dim0, dim1;
   private:   
      size_t _off, _size;
      Tm* _data;
};

// O[l,r,c]
template <typename Tm>   
struct dtensor3{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dim0 & dim1 & dim2 & _off & _size;
      }
      int _addr(const int i0, const int i1, const int i2) const{ 
	 return _off+i0+dim0*(i1+dim1*i2); 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor3(): dim0(0), dim1(0), dim2(0), _off(0), _size(0), _data(nullptr) {}
      void setup_dims(const int _dim0, const int _dim1, const int _dim2, const size_t off){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 _off = off;
	 _size = dim0*dim1*dim2;
      }
      void setup_data(Tm* data){ _data = data; }
      const Tm operator()(const int i0, const int i1, const int i2) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 return _data[_addr(i0,i1,i2)];
      }	     
      Tm& operator()(const int i0, const int i1, const int i2){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 return _data[_addr(i0,i1,i2)];
      }
      size_t size() const{ return _size; };
      Tm* data() const{ return _data+_off; }
      // in-place operation
      void conjugate(){
         std::transform(_data+_off, _data+_off+_size, _data+_off,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      const dtensor2<Tm> get(const int i2) const{
         dtensor2<Tm> dt2;
	 dt2.setup_dims(dim0,dim1,_addr(0,0,i2));
	 dt2.setup_data(_data);
         return dt2;
      }
      dtensor2<Tm> get(const int i2){
         dtensor2<Tm> dt2;
	 dt2.setup_dims(dim0,dim1,_addr(0,0,i2));
	 dt2.setup_data(_data);
         return dt2;
      }
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor3: " << name 
		   << " size=(" << dim0 << "," << dim1 << "," << dim2 << ")" 
		   << std::endl;
	 for(int i2=0; i2<dim2; i2++){
	    (*this).get(i2).print("i2="+std::to_string(i2));
	 } // i2
      }
   public:
      int dim0, dim1, dim2;
   private:
      size_t _off, _size;
      Tm* _data;
};

// O[l,r,c1,c2]
template <typename Tm>   
struct dtensor4{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dim0 & dim1 & dim2 & dim3 & _off & _size;
      }
      int _addr(const int i0, const int i1, const int i2, const int i3) const{ 
	 return _off+i0+dim0*(i1+dim1*(i2+dim2*i3)); 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor4(): dim0(0), dim1(0), dim2(0), dim3(0), _off(0), _size(0), _data(nullptr) {}
      void setup_dims(const int _dim0, const int _dim1, const int _dim2, const int _dim3, const size_t off){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 dim3 = _dim3;
	 _off = off;
	 _size = dim0*dim1*dim2*dim3;
      } 
      void setup_data(Tm* data){ _data = data; }
      const Tm operator()(const int i0, const int i1, const int i2, const int i3) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 assert(i3>=0 && i3<dim3);
	 return _data[_addr(i0,i1,i2,i3)];
      }	     
      Tm& operator()(const int i0, const int i1, const int i2, const int i3){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 assert(i3>=0 && i3<dim3);
	 return _data[_addr(i0,i1,i2,i3)];
      }
      size_t size() const{ return _size; }
      Tm* data() const{ return _data+_off; }
      // in-place operation
      void conjugate(){
         std::transform(_data+_off, _data+_off+_size, _data+_off,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      const dtensor2<Tm> get(const int i2, const int i3) const{
         dtensor2<Tm> dt2;
	 dt2.setup_dims(dim0,dim1,_addr(0,0,i2,i3));
	 dt2.setup_data(_data);
         return dt2;
      }
      dtensor2<Tm> get(const int i2, const int i3){
         dtensor2<Tm> dt2;
	 dt2.setup_dims(dim0,dim1,_addr(0,0,i2,i3));
	 dt2.setup_data(_data);
         return dt2;
      }
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor4: " << name 
		   << " size=(" << dim0 << "," << dim1 << "," << dim2 << "," << dim3 << ")" 
		   << std::endl;
	 for(int i3=0; i3<dim3; i3++){
	    for(int i2=0; i2<dim2; i2++){
	       (*this).get(i2,i3).print("i3,i2="+std::to_string(i3)+","+std::to_string(i2));
	    } // i2
	 } // i3
      }
   public:
      int dim0, dim1, dim2, dim3;
   private:
      size_t _off, _size;
      Tm* _data;
};

} // ctns

#endif
