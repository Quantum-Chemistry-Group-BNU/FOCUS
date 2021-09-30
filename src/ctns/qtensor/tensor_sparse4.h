#ifndef TENSOR_SPARSE4_H
#define TENSOR_SPARSE4_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"

namespace ctns{

template <typename Tm>
struct stensor2;
template <typename Tm>
struct stensor3;

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
      // algebra
      friend stensor4<Tm> operator +(const stensor4<Tm>& qta, const stensor4<Tm>& qtb){
	 assert(qta.info == qtb.info); 
         stensor4<Tm> qt(qta.info);
	 linalg::xcopy(qt.info._size, qta._data, qt._data);
         qt += qtb;
         return qt;
      }
      friend stensor4<Tm> operator -(const stensor4<Tm>& qta, const stensor4<Tm>& qtb){
	 assert(qta.info == qtb.info); 
         stensor4<Tm> qt(qta.info);
	 linalg::xcopy(qt.info._size, qta._data, qt._data);
         qt -= qtb;
         return qt;
      }
      friend stensor4<Tm> operator *(const Tm fac, const stensor4<Tm>& qta){
         stensor4<Tm> qt(qta.info);
	 linalg::xcopy(qt.info._size, qta._data, qt._data);
         qt *= fac;
         return qt;
      }
      friend stensor4<Tm> operator *(const stensor4<Tm>& qt, const Tm fac){
         return fac*qt;
      }
      double normF() const{ return linalg::xnrm2(info._size, _data); }
      // --- SPECIFIC FUNCTIONS ---
      void permCR_signed(); // wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]} 
      // ZL20210413: application of time-reversal operation
      stensor4<Tm> K(const int nbar=0) const;
      // for sweep algorithm
      void from_array(const Tm* array){
         linalg::xcopy(info._size, array, _data);
      }
      void to_array(Tm* array) const{
         linalg::xcopy(info._size, _data, array);
      }
      // for decimation
      inline qproduct dpt_lc1()  const{ return qmerge(info.qrow, info.qmid); };
      inline qproduct dpt_c2r()  const{ return qmerge(info.qver, info.qcol); };
      inline qproduct dpt_c1c2() const{ return qmerge(info.qmid, info.qver); };
      inline qproduct dpt_lr()   const{ return qmerge(info.qrow, info.qcol); };
      // reshape: merge wf4[l,r,c1,c2]
      stensor3<Tm> merge_lc1() const{ // wf3[lc1,r,c2] 
	 auto qprod = dpt_lc1();
	 return merge_qt4_qt3_lc1(*this, qprod.first, qprod.second);
      }
      stensor3<Tm> merge_c2r() const{ // wf3[l,c2r,c1]
	 auto qprod = dpt_c2r();
	 return merge_qt4_qt3_c2r(*this, qprod.first, qprod.second);
      } 
      stensor3<Tm> merge_c1c2() const{ // wf3[l,r,c1c2]
	 auto qprod = dpt_c1c2();
	 return merge_qt4_qt3_c1c2(*this, qprod.first, qprod.second);
      }
      // shorthand function
      // wf4[l,r,c1,c2] => wf3[lc1,r,c2] => wf2[lc1,c2r]
      stensor2<Tm> merge_lc1_c2r() const{
	 return (this->merge_lc1()).merge_cr();
      }
      // wf4[l,r,c1,c2] => wf3[l,r,c1c2] => wf2[lr,c1c2]
      stensor2<Tm> merge_lr_c1c2() const{
	 return (this->merge_c1c2()).merge_lr();
      }
   public:
      bool own = true; // whether the object owns its data
      qinfo4<Tm> info;
   private:   
      Tm* _data = nullptr;
};

// wf[lc1c2r]->wf[lc1c2r]*(-1)^{(p[c1]+p[c2])*p[r]}
template <typename Tm>
void stensor4<Tm>::permCR_signed(){
   int br,bc,bm,bv;
   for(int idx=0; idx<info._qblocks.size(); idx++){
      auto& blk4 = info._qblocks[idx];
      if(blk4.size() == 0) continue;
      info._addr_unpack(idx,br,bc,bm,bv);
      if((info.qmid.get_parity(bm)+info.qver.get_parity(bv))*info.qcol.get_parity(bc) == 1){
         linalg::xscal(blk4.size(), -1.0, blk4.data());
      }
   }
}

// ZL20210510: application of time-reversal operation
template <typename Tm>
stensor4<Tm> stensor4<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   stensor4<Tm> qt4(info.sym.flip(), info.qrow, info.qcol, info.qmid, info.qver);
   int br,bc,bm,bv;
   for(int idx=0; idx<info._qblocks.size(); idx++){
      auto& blk = qt4.info._qblocks[idx];
      if(blk.size() == 0) continue;
      const auto& blkk = info._qblocks[idx];
      info._addr_unpack(idx,br,bc,bm,bv);
      int pr = info.qrow.get_parity(br);
      int pc = info.qcol.get_parity(bc);
      int pm = info.qmid.get_parity(bm);
      int pv = info.qver.get_parity(bv);
      int mdim = info.qmid.get_dim(bm);
      int vdim = info.qver.get_dim(bv);
      // qt4_new(c1c2)[l,r] = qt4(c1c2_bar)[l_bar,r_bar]^*
      if(pm == 0 && pv == 0){
	 for(int iv=0; iv<vdim; iv++){
	    for(int im=0; im<mdim; im++){
	       auto mat = blkk.get(im,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), fpo, mat.data(), blk.get(im,iv).data());
            }
	 }
      }else if(pm == 0 && pv == 1){
	 assert(vdim%2 == 0);
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
	       auto mat = blkk.get(im,iv+vdim2).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), fpo, mat.data(), blk.get(im,iv).data());
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim; im++){
	       auto mat = blkk.get(im,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), -fpo, mat.data(), blk.get(im,iv+vdim2).data());
	    }
	 }
      }else if(pm == 1 && pv == 0){
	 assert(mdim%2 == 0);
	 int mdim2 = mdim/2;
	 for(int iv=0; iv<vdim; iv++){
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im+mdim2,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), fpo, mat.data(), blk.get(im,iv).data());
	    }
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), -fpo, mat.data(), blk.get(im+mdim2,iv).data());
	    }
	 }
      }else if(pm == 1 && pv == 1){
	 assert(mdim%2 == 0 && vdim%2 == 0);
	 int mdim2 = mdim/2;
	 int vdim2 = vdim/2;
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im+mdim2,iv+vdim2).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), fpo, mat.data(), blk.get(im,iv).data());
	    }
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im,iv+vdim2).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), -fpo, mat.data(), blk.get(im+mdim2,iv).data());
	    }
	 }
	 for(int iv=0; iv<vdim2; iv++){
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im+mdim2,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), -fpo, mat.data(), blk.get(im,iv+vdim2).data());
	    }
	    for(int im=0; im<mdim2; im++){
	       auto mat = blkk.get(im,iv).time_reversal(pr, pc);
	       linalg::xaxpy(mat.size(), fpo, mat.data(), blk.get(im+mdim2,iv+vdim2).data());
	    }
	 }
      } // (pm,pv)
   } // idx
   return qt4;
}

} // ctns

#endif
