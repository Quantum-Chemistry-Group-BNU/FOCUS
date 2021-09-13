#ifndef TENSOR_SPARSE3_H
#define TENSOR_SPARSE3_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

template <typename Tm>
struct stensor3{
   public:
      // --- GENERAL FUNCTIONS ---
      // constructors
      stensor3(): _data(nullptr) {}
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
	        const std::vector<bool> _dir={0,1,1}){
         info.init(_sym, _qrow, _qcol, _qmid, _dir);
         _data = new Tm[info._size];
	 info.setup_data(_data);
      }
      stensor3(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
	       const std::vector<bool> _dir={0,1,1}){
	 this->init(_sym, _qrow, _qcol, _qmid, _dir);     
      }
      // desctructors
      ~stensor3(){ delete[] _data; }
      stensor3(const stensor3& st) = delete;
      stensor3& operator =(const stensor3& st) = delete;
      /*
      // copy constructor
      stensor3(const stensor3& st){
	 std::cout << "stensor3: copy constructor" << std::endl;     
         info = st.info;
	 _data = new Tm[info._size];
	 std::copy_n(st._data, info._size, _data);
	 info.setup_data(_data);
      }
      // copy assignment
      stensor3& operator =(const stensor3& st){
	 std::cout << "stensor3: copy assignment" << std::endl;     
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
      stensor3(stensor3&& st){
	 std::cout << "stensor3: move constructor" << std::endl;     
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor3& operator =(stensor3&& st){
	 std::cout << "stensor3: move assignment" << std::endl;     
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
      int row_dim(const int br) const{ return info.qrow.get_dim(br); }
      int col_dim(const int bc) const{ return info.qcol.get_dim(bc); } 
      int mid_dim(const int bm) const{ return info.qmid.get_dim(bm); }
      int row_dimAll() const{ return info.qrow.get_dimAll(); }
      int col_dimAll() const{ return info.qcol.get_dimAll(); } 
      int mid_dimAll() const{ return info.qmid.get_dimAll(); }
      size_t size() const{ return info._size; }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      dtensor3<Tm>& operator ()(const int br, const int bc, const int bm){ 
	 return info._qblocks[info._addr(br,bc,bm)]; 
      }
      const dtensor3<Tm>& operator ()(const int br, const int bc, const int bm) const{ 
	 return info._qblocks[info._addr(br,bc,bm)]; 
      }
      // --- SPECIFIC FUNCTIONS ---
   public:
      qinfo3<Tm> info;
   private:   
      Tm* _data;
};

} // ctns

#endif
