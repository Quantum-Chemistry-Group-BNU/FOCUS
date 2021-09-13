#ifndef TENSOR_SPARSE4_H
#define TENSOR_SPARSE4_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

template <typename Tm>
struct stensor4{
   public:
      // --- GENERAL FUNCTIONS ---
      // constructors
      stensor4(): _data(nullptr) {};
      void init(const qsym& _sym, 
	        const qbond& _qrow, const qbond& _qcol, 
	        const qbond& _qmid, const qbond& _qver){
         info.init(_sym, _qrow, _qcol, _qmid, _qver);
         _data = new Tm[info._size];
	 info.setup_data(_data);
      }
      stensor4(const qsym& _sym, 
	       const qbond& _qrow, const qbond& _qcol, 
	       const qbond& _qmid, const qbond& _qver){
         this->init(_sym, _qrow, _qcol, _qmid, _qver);
      }
      // desctructors
      ~stensor4(){ delete[] _data; }
      stensor4(const stensor4& st) = delete;
      stensor4& operator =(const stensor4& st) = delete;
      /*
      // copy constructor
      stensor4(const stensor4& st){
	 //std::cout << "stensor4: copy constructor" << std::endl;     
         info = st.info;
	 _data = new Tm[info._size];
	 std::copy_n(st._data, info._size, _data);
	 info.setup_data(_data);
      }
      // copy assignment
      stensor4& operator =(const stensor4& st){
	 //std::cout << "stensor4: copy assignment" << std::endl;     
         if(this != &st){
            info = st.info;
	    delete[] _data;
	    _data = new Tm[info._size];
	    std::copy_n(st._data, info._size, _data);
	    info.setup_data(_data);
	 }
	 return *this;
      }
      */
      // move constructor
      stensor4(stensor4&& st){
	 //std::cout << "stensor4: move constructor" << std::endl;     
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor4& operator =(stensor4&& st){
	 //std::cout << "stensor4: move assignment" << std::endl;     
         if(this != &st){
            info = std::move(st.info);
	    delete[] _data;
	    _data = st._data;
            st._data = nullptr;	    
	 }
	 return *this;
      }
      // helpers
      int rows() const{ return info._rows; }
      int cols() const{ return info._cols; }
      int mids() const{ return info._mids; }
      int vers() const{ return info._vers; }
      int row_dim(const int br) const{ return info.qrow.get_dim(br); }
      int col_dim(const int bc) const{ return info.qcol.get_dim(bc); } 
      int mid_dim(const int bm) const{ return info.qmid.get_dim(bm); }
      int ver_dim(const int bv) const{ return info.qver.get_dim(bv); }
      int row_dimAll() const{ return info.qrow.get_dimAll(); }
      int col_dimAll() const{ return info.qcol.get_dimAll(); } 
      int mid_dimAll() const{ return info.qmid.get_dimAll(); }
      int ver_dimAll() const{ return info.qver.get_dimAll(); }
      size_t size() const{ return info._size; }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      dtensor4<Tm>& operator ()(const int br, const int bc, const int bm, const int bv){ 
	 return info._qblocks[info._addr(br,bc,bm,bv)]; 
      }
      const dtensor4<Tm>& operator ()(const int br, const int bc, const int bm, const int bv) const{ 
	 return info._qblocks[info._addr(br,bc,bm,bv)]; 
      }
      // --- SPECIFIC FUNCTIONS ---
   public:
      qinfo4<Tm> info;
   private:   
      Tm* _data;
};

} // ctns

#endif
