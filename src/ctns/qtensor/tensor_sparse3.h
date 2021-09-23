#ifndef TENSOR_SPARSE3_H
#define TENSOR_SPARSE3_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

template <typename Tm>
struct stensor2;

const bool debug_sparse3 = false;
extern const bool debug_sparse3;

template <typename Tm>
struct stensor3{
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
      stensor3(){}
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
	        const std::vector<bool> _dir={0,1,1}, const bool _own=true){
         info.init(_sym, _qrow, _qcol, _qmid, _dir);
	 own = _own;
	 if(own) this->allocate();
      }
      void setup_data(Tm* data){
         assert(own == false);
	 _data = data;
	 info.setup_data(_data);
      }
      stensor3(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
	       const std::vector<bool> _dir={0,1,1}, const bool _own=true){
	 this->init(_sym, _qrow, _qcol, _qmid, _dir, _own);
      }
      // simple constructor from qinfo
      stensor3(const qinfo3<Tm>& _info, const bool _own=true){
	 info = _info;
	 own = _own;
         if(own) this->allocate();
      }
      // desctructors
      ~stensor3(){ 
	 if(own) delete[] _data; 
      }
      // copy constructor
      stensor3(const stensor3& st) = delete;
      // copy assignment
      stensor3& operator =(const stensor3& st) = delete;
/*
      // copy constructor
      stensor3(const stensor3& st){
	 std::cout << "stensor3: copy constructor" << std::endl;     
         info = st.info;
	 _data = new Tm[info._size];
	 linalg::xcopy(info._size, st._data, _data);
	 info.setup_data(_data);
      }
      // copy assignment
      stensor3& operator =(const stensor3& st){
	 std::cout << "stensor3: copy assignment" << std::endl;    
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
      stensor3(stensor3&& st){
	 if(debug_sparse3) std::cout << "stensor3: move constructor - st.own=" << st.own << std::endl;    
	 assert(own == true);
	 own = st.own; 
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor3& operator =(stensor3&& st){
	 if(debug_sparse3) std::cout << "stensor3: move assignment - st.own=" << st.own << std::endl;
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
      int row_dimAll() const{ return info.qrow.get_dimAll(); }
      int col_dimAll() const{ return info.qcol.get_dimAll(); } 
      int mid_dimAll() const{ return info.qmid.get_dimAll(); }
      int row_dim(const int br) const{ return info.qrow.get_dim(br); }
      int col_dim(const int bc) const{ return info.qcol.get_dim(bc); } 
      int mid_dim(const int bm) const{ return info.qmid.get_dim(bm); }
      qsym row_sym(const int br) const{ return info.qrow.get_sym(br); }
      qsym col_sym(const int bc) const{ return info.qcol.get_sym(bc); } 
      qsym mid_sym(const int bm) const{ return info.qmid.get_sym(bm); }
      bool dir_row() const{ return info.dir[0]; } 
      bool dir_col() const{ return info.dir[1]; }
      bool dir_mid() const{ return info.dir[2]; } 
      size_t size() const{ return info._size; }
      Tm* data() const{ return _data; }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      const dtensor3<Tm> operator()(const int br, const int bc, const int bm) const{ 
	 return info._qblocks[info._addr(br,bc,bm)]; 
      }
      dtensor3<Tm>& operator()(const int br, const int bc, const int bm){ 
	 return info._qblocks[info._addr(br,bc,bm)]; 
      }
      // simple arithmetic operations
      stensor3<Tm>& operator *=(const Tm fac){
         linalg::xscal(info._size, fac, _data);
         return *this;
      }
      stensor3<Tm>& operator +=(const stensor3<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, 1.0, st.data(), _data);
         return *this;
      }
      stensor3<Tm>& operator -=(const stensor3<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, -1.0, st.data(), _data);
         return *this;
      }
      // --- SPECIFIC FUNCTIONS ---
      // fix middle index (bm,im) - bm-th block, im-idx - composite index!
      stensor2<Tm> fix_mid(const std::pair<int,int> mdx) const;
      // deal with fermionic sign in fermionic direct product
      void mid_signed(const double fac=1.0); // wf[lcr](-1)^{p(c)}
      void row_signed(const double fac=1.0); // wf[lcr](-1)^{p(l)}
      void permCR_signed(); // wf[lcr]->wf[lcr]*(-1)^{p[c]*p[r]}
      // for sweep algorithm
      void from_array(const Tm* array){
	 linalg::xcopy(info._size, array, _data);
      }
      void to_array(Tm* array) const{
         linalg::xcopy(info._size, _data, array);
      }
/*
      // for decimation
      qproduct dpt_lc() const{ return qmerge(info.qrow, info.qmid); }
      qproduct dpt_cr() const{ return qmerge(info.qmid, info.qcol); }
      qproduct dpt_lr() const{ return qmerge(info.qrow, info.qcol); }
      // reshape: merge
      stensor2<Tm> merge_lc() const{ 
	 auto qprod = dpt_lc();
	 return merge_qt3_qt2_lc(*this, qprod.first, qprod.second);
      }
      stensor2<Tm> merge_cr() const{
	 auto qprod = dpt_cr(); 
	 return merge_qt3_qt2_cr(*this, qprod.first, qprod.second);
      }
      stensor2<Tm> merge_lr() const{
	 auto qprod = dpt_lr();  
	 return merge_qt3_qt2_lr(*this, qprod.first, qprod.second);
      }
*/
   public:
      bool own = true; // whether the object owns its data
      qinfo3<Tm> info;
   private:   
      Tm* _data = nullptr;
};

// fix middle index (bm,im) - bm-th block, im-idx - composite index!
// A(l,r) = B[m](l,r)
template <typename Tm>
stensor2<Tm> stensor3<Tm>::fix_mid(const std::pair<int,int> mdx) const{
   int bm = mdx.first, im = mdx.second;   
   assert(im < info.qmid.get_dim(bm));
   auto symIn = info.dir[2] ? info.sym-info.qmid.get_sym(bm) : info.sym+info.qmid.get_sym(bm);
   stensor2<Tm> qt2(symIn, info.qrow, info.qcol, {info.dir[0], info.dir[1]});
   for(int br=0; br<info._rows; br++){
      for(int bc=0; bc<info._cols; bc++){
	 const auto& blk3 = (*this)(br,bc,bm);
         if(blk3.size() == 0) continue;
         auto& blk2 = qt2(br,bc);
	 int N = blk2.size();
	 linalg::xcopy(N, blk3.get(im).data(), blk2.data()); 
      } // bc
   } // br
   return qt2;
}

// deal with fermionic sign in fermionic direct product
// wf[lcr](-1)^{p(c)}
template <typename Tm>
void stensor3<Tm>::mid_signed(const double fac){
   int br,bc,bm;
   for(int idx=0; idx<info._qblocks.size(); idx++){
      auto& blk3 = info._qblocks[idx];
      if(blk3.size() == 0) continue;
      info._addr_unpack(idx,br,bc,bm);
      double fac2 = (info.qmid.get_parity(bm)==0)? fac : -fac;
      linalg::xscal(blk3.size(), fac2, blk3.data());  
   }
}

// wf[lcr](-1)^{p(l)}
template <typename Tm>
void stensor3<Tm>::row_signed(const double fac){
   int br,bc,bm;
   for(int idx=0; idx<info._qblocks.size(); idx++){
      auto& blk3 = info._qblocks[idx];
      if(blk3.size() == 0) continue;
      info._addr_unpack(idx,br,bc,bm);
      double fac2 = (info.qrow.get_parity(br)==0)? fac : -fac;
      linalg::xscal(blk3.size(), fac2, blk3.data());  
   }
}

// Generate the sign for wf[lcr]|lcr> = wf3[lcr]|lrc> 
// with wf3[lcr] = wf[lcr]*(-1)^{p[c]*p[r]}|lrc>
// which is later used for wf3[l,c,r] <-> wf2[lr,c] (merge_lr)
template <typename Tm>
void stensor3<Tm>::permCR_signed(){
   int br,bc,bm;
   for(int idx=0; idx<info._qblocks.size(); idx++){
      auto& blk3 = info._qblocks[idx];
      if(blk3.size() == 0) continue;
      info._addr_unpack(idx,br,bc,bm);
      if(info.qmid.get_parity(bm)*info.qcol.get_parity(bc) == 1){
         linalg::xscal(blk3.size(), -1.0, blk3.data());
      }
   }
}

} // ctns

#endif
