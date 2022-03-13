#ifndef DTENSOR_H
#define DTENSOR_H

#include <cassert>
#include "../../core/serialization.h"
#include "../../core/matrix.h"
#include "../../core/linalg.h"

namespace ctns{

//
// Dense tensor stored in FORTRAN ORDER to interface with BLAS/LAPACK
// They are just wrapper for data, so they are free to be copied/moved. 
// Move: e.g., xgemm("N","T",1.0,blk3a.get(im),blk2b,1.0,blk3.get(im));
//
	
// O[l,r]
template <typename Tm>   
struct dtensor2 : public linalg::BaseMatrix<Tm> {
   private:
      int _addr(const int i0, const int i1) const{ 
	 return i0+dim0*i1; 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor2(){}
      dtensor2(const int _dim0, 
	       const int _dim1,
	       Tm* data){
	 dim0 = _dim0;
	 dim1 = _dim1; 
	 _data = data;
	 _size = dim0*dim1;
      }
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
      size_t size() const{ return _size; }
      const Tm* data() const{ return _data; }
      Tm* data(){ return _data; }
      bool empty() const{ return _data==nullptr; }
      // in-place operation
      void clear(){ memset(_data, 0, _size*sizeof(Tm)); }
      void conjugate(){
         std::transform(_data, _data+_size, _data,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor2: " << name 
		   << " size=(" << dim0 << "," << dim1 << ")"
                   << " _data=" << _data 
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
	 linalg::matrix<Tm> mat(dim0,dim1,_data);
	 return mat;
      }
      linalg::matrix<Tm> time_reversal(const int pr, const int pc) const;
      // interface with xgemm, similar to linalg::matrix 
      int rows() const{ return dim0; }
      int cols() const{ return dim1; }
      const Tm* col(const int j) const{ return &_data[_addr(0,j)]; }; 
      Tm* col(const int j){ return &_data[_addr(0,j)]; }; 
   public:
      Tm* _data = nullptr;
      int dim0 = 0, dim1 = 0; 
      int _size = 0;
};

// O[l,r,c]
template <typename Tm>   
struct dtensor3{
   private:
      int _addr(const int i0, const int i1, const int i2) const{ 
	 return i0+dim0*(i1+dim1*i2); 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor3(){}
      dtensor3(const int _dim0, 
	       const int _dim1, 
	       const int _dim2,
	       Tm* data){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 _data = data;
	 _size = dim0*dim1*dim2;
      }
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
      size_t size() const{ return _size; }
      const Tm* data() const{ return _data; }
      Tm* data(){ return _data; }
      bool empty() const{ return _data==nullptr; }
      // in-place operation
      void clear(){ memset(_data, 0, _size*sizeof(Tm)); } 
      void conjugate(){
         std::transform(_data, _data+_size, _data,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      const dtensor2<Tm> get(const int i2) const{
         return dtensor2<Tm>(dim0,dim1,_data+_addr(0,0,i2));
      }
      dtensor2<Tm> get(const int i2){
         return dtensor2<Tm>(dim0,dim1,_data+_addr(0,0,i2));
      }
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor3: " << name 
		   << " size=(" << dim0 << "," << dim1 << "," << dim2 << ")" 
                   << " _data=" << _data 
		   << std::endl;
	 for(int i2=0; i2<dim2; i2++){
	    const auto dt2 = this->get(i2);
	    dt2.print("i2="+std::to_string(i2),prec);
	 } // i2
      }
   public:
      Tm* _data = nullptr;
      int dim0 = 0, dim1 = 0, dim2 = 0;
      int _size = 0;
};

// O[l,r,c1,c2]
template <typename Tm>   
struct dtensor4{
   private:
      int _addr(const int i0, const int i1, const int i2, const int i3) const{ 
	 return i0+dim0*(i1+dim1*(i2+dim2*i3)); 
      }
   public:
      // --- GENERAL FUNCTIONS ---
      dtensor4(){}
      dtensor4(const int _dim0, 
	       const int _dim1, 
	       const int _dim2, 
	       const int _dim3,
	       Tm* data){
	 dim0 = _dim0;
	 dim1 = _dim1;
	 dim2 = _dim2;
	 dim3 = _dim3;
	 _data = data;
	 _size = dim0*dim1*dim2*dim3;
      } 
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
      const Tm* data() const{ return _data; }
      Tm* data(){ return _data; }
      bool empty() const{ return _data==nullptr; }
      // in-place operation
      void clear(){ memset(_data, 0, _size*sizeof(Tm)); }
      void conjugate(){
         std::transform(_data, _data+_size, _data,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // --- SPECIFIC FUNCTIONS ---
      const dtensor2<Tm> get(const int i2, const int i3) const{
	 return dtensor2<Tm>(dim0,dim1,_data+_addr(0,0,i2,i3));
      }
      dtensor2<Tm> get(const int i2, const int i3){
	 return dtensor2<Tm>(dim0,dim1,_data+_addr(0,0,i2,i3));
      }
      const dtensor3<Tm> get(const int i3) const{
         return dtensor3<Tm>(dim0,dim1,dim2,_data+_addr(0,0,0,i3));
      }
      dtensor3<Tm> get(const int i3){
         return dtensor3<Tm>(dim0,dim1,dim2,_data+_addr(0,0,0,i3));
      }
      // print
      void print(std::string name="", const int prec=4) const{
         std::cout << " dtensor4: " << name 
		   << " size=(" << dim0 << "," << dim1 << "," << dim2 << "," << dim3 << ")" 
                   << " _data=" << _data 
		   << std::endl;
	 for(int i3=0; i3<dim3; i3++){
	    for(int i2=0; i2<dim2; i2++){
	       const auto dt2 = this->get(i2,i3);
	       dt2.print("i2,i3="+std::to_string(i2)+","+std::to_string(i3),prec);
	    } // i2
	 } // i3
      }
   public:
      Tm* _data = nullptr;
      int dim0 = 0, dim1 = 0, dim2 = 0, dim3 = 0;
      int _size = 0;
};

// 
// Functions
//

// M(l,r) = M1(bar{l},bar{r})^* given parity of qr and qc
template <typename Tm>
linalg::matrix<Tm> dtensor2<Tm>::time_reversal(const int pr, const int pc) const{
   const int& _rows = dim0;
   const int& _cols = dim1;
   linalg::matrix<Tm> blk(_rows,_cols);
   // even-even block:
   //    <e|\bar{O}|e> = p{O} <e|O|e>^*
   if(pr == 0 && pc == 0){
      std::transform(this->data(), this->data()+this->size(), blk.data(),
		     [](const Tm& x){ return tools::conjugate(x); });
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

} // ctns

#endif
