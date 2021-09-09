#ifndef TENSOR_DENSE_H
#define TENSOR_DENSE_H

#include <cassert>
#include "../../core/serialization.h"

namespace ctns{

//
// Dense tensor stored in Fortran order
//
	
// O[l,r]
template <typename Tm>   
struct dtensor2{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dim0 & dim1 & _off & _size;
      }
   public:
      dtensor2(): dim0(0), dim1(0), _off(0), _size(0), _data(nullptr) {}
      void setup_dims(const int _dim0, const int _dim1, const size_t off){
	 dim0 = _dim0;
	 dim1 = _dim1; 
	 _off = off;
	 _size = dim0*dim1; 
      }
      void setup_data(const Tm* data){ _data = data; }
      const Tm operator()(const int i0, const int i1) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 return _data[_off+i0+dim0*i1];
      }
      Tm& operator()(const int i0, const int i1){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 return _data[_off+i0+dim0*i1];
      } 
      size_t size() const{ return _size; }; 
   public:
      int dim0, dim1;
   private:   
      size_t _off, _size;
      Tm* _data;
};

/*
// O[l,r,c]
template <typename Tm>   
struct dtensor3{
   public:
      dtensor3(): dim0(0), dim1(0), dim2(0), data(nullptr) {}
      void setup(const int _dim0, const int _dim1, const int _dim2, Tm* _data){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 data = _data;
	 size = dim0*dim1*dim2;
      }
      const Tm operator()(const int i0, const int i1, const int i2) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 return data[i0+dim0*(i1+dim1*i2)];
      }	     
      Tm& operator()(const int i0, const int i1, const int i2){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 return data[i0+dim0*(i1+dim1*i2)];
      }
   public:
      size_t size = 0;
      int dim0,dim1,dim2;
      Tm* data;
};

// O[l,r,c1,c2]
template <typename Tm>   
struct dtensor4{
   public:
      dtensor4(): dim0(0), dim1(0), dim2(0), dim3(0), data(nullptr) {}
      void setup(const int _dim0, const int _dim1, const int _dim2, const int _dim3, Tm* _data){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 dim3 = _dim3;
	 data = _data;
	 size = dim0*dim1*dim2*dim3;
      } 
      const Tm operator()(const int i0, const int i1, const int i2, const int i3) const{
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 assert(i3>=0 && i3<dim3);
	 return data[i0+dim0*(i1+dim1*(i2+dim2*i3))];
      }	     
      Tm& operator()(const int i0, const int i1, const int i2, const int i3){ 
         assert(i0>=0 && i0<dim0);
	 assert(i1>=0 && i1<dim1);
	 assert(i2>=0 && i2<dim2);
	 assert(i3>=0 && i3<dim3);
	 return data[i0+dim0*(i1+dim1*(i2+dim2*i3))];
      }
   public:
      size_t size = 0;
      int dim0,dim1,dim2,dim3;
      Tm* data;
};
*/

} // ctns

#endif
