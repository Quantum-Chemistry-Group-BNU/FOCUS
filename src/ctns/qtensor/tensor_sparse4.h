#ifndef TENSOR_SPARSE4_H
#define TENSOR_SPARSE4_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

const bool debug_sparse4 = false;
extern const bool debug_sparse4;

template <typename Tm>
struct stensor4{
   private:
      friend class boost::serialization::access;	   
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & own & info;
	 if(own){
            for(int i=0; i<info._size; i++){
	       ar & _data[i];
	    }
	 }
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & own & info;
	 if(own){
	    _data = new Tm[info._size];
            for(int i=0; i<info._size; i++){
	       ar & _data[i];
	    }
	    info.setup_data(_data);
	 }
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
      // memory allocation
      void allocate(){
         _data = new Tm[info._size];
	 memset(_data, 0, info._size*sizeof(Tm));
	 info.setup_data(_data);
      }
   public:
      // --- GENERAL FUNCTIONS ---
      // constructors
      stensor4(){}
      void init(const qsym& _sym, 
	        const qbond& _qrow, const qbond& _qcol, 
	        const qbond& _qmid, const qbond& _qver,
		const bool _own=true){
         info.init(_sym, _qrow, _qcol, _qmid, _qver);
	 own = _own;
	 if(own) this->allocate();
      }
      void setup_data(Tm* data){
         assert(own == false);
	 _data = data;
	 info.setup_data(_data);
      }
      stensor4(const qsym& _sym, 
	       const qbond& _qrow, const qbond& _qcol, 
	       const qbond& _qmid, const qbond& _qver,
	       const bool _own=true){
         this->init(_sym, _qrow, _qcol, _qmid, _qver, _own);
      }
      // simple constructor from qinfo
      stensor4(const qinfo4<Tm>& _info, const bool _own=true){
	 info = _info;
	 own = _own;
         if(own) this->allocate();
      }
      // desctructors
      ~stensor4(){ 
	 if(own) delete[] _data; 
      }
      // copy constructor
      stensor4(const stensor4& st) = delete;
      // copy assignment
      stensor4& operator =(const stensor4& st) = delete;
/*
      // copy constructor
      stensor4(const stensor4& st){
	 std::cout << "stensor4: copy constructor" << std::endl;     
         info = st.info;
	 _data = new Tm[info._size];
	 linalg::xcopy(info._size, st._data, _data);
	 info.setup_data(_data);
      }
      // copy assignment
      stensor4& operator =(const stensor4& st){
	 std::cout << "stensor4: copy assignment" << std::endl;     
	 exit(1); 
         if(this != &st){
            info = st.info;
	    delete[] _data;
	    _data = new Tm[info._size];
	    linalg::xcopy(info._size, st._data, _data);
	    info.setup_data(_data);
	 }
	 return *this;
      }
*/
      // move constructor
      stensor4(stensor4&& st){
	 if(debug_sparse4) std::cout << "stensor4: move constructor - st.own=" << st.own << std::endl;
	 assert(own == true);
	 own = st.own; 
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor4& operator =(stensor4&& st){
	 if(debug_sparse4) std::cout << "stensor4: move assignment - st.own=" << st.own << std::endl;     
	 assert(own == true && st.own == true); 
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
      int row_dimAll() const{ return info.qrow.get_dimAll(); }
      int col_dimAll() const{ return info.qcol.get_dimAll(); } 
      int mid_dimAll() const{ return info.qmid.get_dimAll(); }
      int ver_dimAll() const{ return info.qver.get_dimAll(); }
      int row_dim(const int br) const{ return info.qrow.get_dim(br); }
      int col_dim(const int bc) const{ return info.qcol.get_dim(bc); } 
      int mid_dim(const int bm) const{ return info.qmid.get_dim(bm); }
      int ver_dim(const int bv) const{ return info.qver.get_dim(bv); }
      qsym row_sym(const int br) const{ return info.qrow.get_sym(br); }
      qsym col_sym(const int bc) const{ return info.qcol.get_sym(bc); } 
      qsym mid_sym(const int bm) const{ return info.qmid.get_sym(bm); }
      qsym ver_sym(const int bv) const{ return info.qver.get_sym(bv); }
      bool dir_row() const{ return info.dir[0]; } 
      bool dir_col() const{ return info.dir[1]; }
      bool dir_mid() const{ return info.dir[2]; } 
      bool dir_ver() const{ return info.dir[3]; } 
      size_t size() const{ return info._size; }
      Tm* data() const{ return _data; }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      const dtensor4<Tm> operator()(const int br, const int bc, const int bm, const int bv) const{ 
	 return info._qblocks[info._addr(br,bc,bm,bv)]; 
      }
      dtensor4<Tm>& operator()(const int br, const int bc, const int bm, const int bv){ 
	 return info._qblocks[info._addr(br,bc,bm,bv)]; 
      }
      // simple arithmetic operations
      stensor4<Tm>& operator *=(const Tm fac){
         linalg::xscal(info._size, fac, _data);
         return *this;
      }
      stensor4<Tm>& operator +=(const stensor4<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, 1.0, st.data(), _data);
         return *this;
      }
      stensor4<Tm>& operator -=(const stensor4<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, -1.0, st.data(), _data);
         return *this;
      }
      // --- SPECIFIC FUNCTIONS ---
      // for sweep algorithm
      void from_array(const Tm* array){
         linalg::xcopy(info._size, array, _data);
      }
      void to_array(Tm* array) const{
         linalg::xcopy(info._size, _data, array);
      }
   public:
      bool own = true; // whether the object owns its data
      qinfo4<Tm> info;
   private:   
      Tm* _data = nullptr;
};

} // ctns

#endif
