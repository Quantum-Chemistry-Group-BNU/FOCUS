#ifndef STENSOR2_H
#define STENSOR2_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"
#include "../../core/linalg.h"

namespace ctns{

template <typename Tm>
struct stensor3;
template <typename Tm>
struct stensor4;

const bool debug_stensor2 = false;
extern const bool debug_stensor2; 

template <typename Tm>
struct stensor2{
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
	 info.setup_data(_data);
	 memset(_data, 0, info._size*sizeof(Tm));
      }
   public:
      // --- GENERAL FUNCTIONS ---
      // constructors
      stensor2(){};
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
	        const std::vector<bool> _dir={1,0}, const bool _own=true){
         info.init(_sym, _qrow, _qcol, _dir);
         own = _own;
         if(own) this->allocate();
      }
      stensor2(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
	       const std::vector<bool> _dir={1,0}, const bool _own=true){
	 this->init(_sym, _qrow, _qcol, _dir, _own);
      }
      // simple constructor from qinfo
      void init(const qinfo2<Tm>& _info, const bool _own=true){
	 info = _info;
	 own = _own;
         if(own) this->allocate();
      }
      stensor2(const qinfo2<Tm>& _info, const bool _own=true){
	 this->init(_info, _own);
      }
      // used to for setup ptr, if own=false
      void setup_data(Tm* data){
         assert(own == false);
	 _data = data;
	 info.setup_data(_data);
      }
      // desctructors
      ~stensor2(){ 
	 if(own) delete[] _data; 
      }
      // copy constructor 
      stensor2(const stensor2& st){
	 if(debug_stensor2) std::cout << "stensor2: copy constructor - st.own=" << st.own << std::endl;   
	 //assert(st.own == false);
	 own = st.own;
	 info = st.info;
	 if(st.own){
	    _data = new Tm[info._size];
	    info.setup_data(_data);
	    linalg::xcopy(info._size, st._data, _data);
	 }else{
	    // shalow copy of the wrapper in case st.own = false;
	    // needs to be here for direct manipulations of data in xaxpy
	    _data = st._data; 
	 }
      }
      // copy assignment
      stensor2& operator =(const stensor2& st) = delete;
/*
      // copy assignment
      stensor2& operator =(const stensor2& st){
	 std::cout << "stensor2: copy assignment - exit" << std::endl;    
	 exit(1); 
         if(this != &st){
            info = st.info;
	    delete[] _data;
	    _data = new Tm[info._size];
	    info.setup_data(_data);
	    linalg::xcopy(info._size, st._data, _data);
	 }
	 return *this;
      }
*/
      // move constructor
      stensor2(stensor2&& st){
	 if(debug_stensor2) std::cout << "stensor2: move constructor - st.own=" << st.own << std::endl;     
	 assert(own == true);
	 own = st.own;
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor2& operator =(stensor2&& st){
	 if(debug_stensor2) std::cout << "stensor2: move assignment - st.own=" << st.own << std::endl;    
	 // only move if the data is owned by the object, 
	 // otherwise data needs to be copied explicitly!
	 // e.g., linalg::xcopy(info._size, st._data, _data);
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
      bool dir_row() const{ return info.dir[0]; } 
      bool dir_col() const{ return info.dir[1]; } 
      size_t size() const{ return info._size; }
      Tm* data() const{ return _data; }
      // in-place operation
      void conjugate(){
         std::transform(_data, _data+info._size, _data,
			[](const Tm& x){ return tools::conjugate(x); });
      }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      const dtensor2<Tm> operator()(const int br, const int bc) const{ 
	 return info._qblocks[info._addr(br,bc)]; 
      }
      dtensor2<Tm>& operator()(const int br, const int bc){ 
	 return info._qblocks[info._addr(br,bc)]; 
      }
      // simple arithmetic operations
      stensor2<Tm>& operator *=(const Tm fac){
         linalg::xscal(info._size, fac, _data);
         return *this;
      }
      stensor2<Tm>& operator +=(const stensor2<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, 1.0, st.data(), _data);
         return *this;
      }
      stensor2<Tm>& operator -=(const stensor2<Tm>& st){
	 assert(info == st.info);
	 linalg::xaxpy(info._size, -1.0, st.data(), _data);
         return *this;
      }
      stensor2<Tm> operator -() const{
         stensor2<Tm> st(info);
	 linalg::xaxpy(info._size, -1.0, _data, st._data);
	 return st;
      }
      // algebra
      friend stensor2<Tm> operator +(const stensor2<Tm>& qta, const stensor2<Tm>& qtb){
	 assert(qta.info == qtb.info); 
         stensor2<Tm> qt(qta.info);
	 linalg::xcopy(qt.info._size, qta._data, qt._data);
         qt += qtb;
         return qt;
      }
      friend stensor2<Tm> operator -(const stensor2<Tm>& qta, const stensor2<Tm>& qtb){
	 assert(qta.info == qtb.info); 
         stensor2<Tm> qt(qta.info);
	 linalg::xcopy(qt.info._size, qta._data, qt._data);
         qt -= qtb;
         return qt;
      }
      friend stensor2<Tm> operator *(const Tm fac, const stensor2<Tm>& qta){
         stensor2<Tm> qt(qta.info);
	 linalg::xaxpy(qt.info._size, fac, qta._data, qt._data);
         return qt;
      }
      friend stensor2<Tm> operator *(const stensor2<Tm>& qt, const Tm fac){
         return fac*qt;
      }
      double normF() const{ return linalg::xnrm2(info._size, _data); }
      void clear(){ memset(_data, 0, info._size*sizeof(Tm)); }
      // --- SPECIFIC FUNCTIONS ---
      stensor2<Tm> view(const bool ifdagger=false) const;
      // from/to dense matrix: assign block to proper place
      void from_matrix(const linalg::matrix<Tm>& mat); 
      linalg::matrix<Tm> to_matrix() const;
      // check whether <l|o|r> is a faithful rep for o=I
      double check_identityMatrix(const double thresh_ortho, const bool debug=false) const;
      // algebra
      stensor2<Tm> dot(const stensor2<Tm>& qt) const{ return contract_qt2_qt2(*this, qt); }
      // ZL20200531: Permute the line of diagrams, while maintaining their directions
      // 	     This does not change the tensor, but just permute order of index
      stensor2<Tm> T() const;
      // ZL20200531: This is used in taking Hermitian conjugate of operators.
      // 	     If row/col is permuted while dir fixed, this effectively changes 
      // 	     the direction of lines in diagrams
      stensor2<Tm> H() const;
      // ZL20210401: generate matrix representation for Kramers paired operators
      stensor2<Tm> K(const int nbar=0) const;
      // for sweep algorithm
      void add_noise(const double noise){
         auto rand = linalg::random_matrix<Tm>(info._size,1);
	 linalg::xaxpy(info._size, noise, rand.data(), _data);
      }
      // reshape: split into stensor3
      // wf2[lc,r] -> wf3[l,r,c]
      stensor3<Tm> split_lc(const qbond& qlx, const qbond& qcx) const{
         auto dpt = qmerge(qlx, qcx).second;
         return split_qt3_qt2_lc(*this, qlx, qcx, dpt);
      }
      // wf2[l,cr] -> wf3[l,r,c]
      stensor3<Tm> split_cr(const qbond& qcx, const qbond& qrx) const{
	 auto dpt = qmerge(qcx, qrx).second;
         return split_qt3_qt2_cr(*this, qcx, qrx, dpt);
      }
      // wf2[lr,c] -> wf3[l,r,c]
      stensor3<Tm> split_lr(const qbond& qlx, const qbond& qrx) const{
	 auto dpt = qmerge(qlx, qrx).second;
         return split_qt3_qt2_lr(*this, qlx, qrx, dpt);
      }
      // shorthand function
      // wf2[lr,c1c2] => wf3[l,r,c1c2] => wf4[l,r,c1,c2] 
      stensor4<Tm> split_lr_c1c2(const qbond& qlx, const qbond& qrx, const qbond& qc1, const qbond& qc2) const{
         return (this->split_lr(qlx, qrx)).split_c1c2(qc1, qc2);
      }
   public:
      bool own = true; // whether the object owns its data
      qinfo2<Tm> info;
   private:
      Tm* _data = nullptr;
};

// ZL@20220221: just view instead of H()
template <typename Tm>
stensor2<Tm> stensor2<Tm>::view(const bool ifdagger) const{
   stensor2<Tm> qt2;
   if(!ifdagger){
      qt2.init(info, false);
      qt2.setup_data(_data);
   }else{
      qt2.init(-info.sym, info.qcol, info.qrow, info.dir, false);
      int br, bc;
      for(int i=0; i<qt2.info._nnzaddr.size(); i++){
         int addr = qt2.info._nnzaddr[i];
         qt2.info._addr_unpack(addr,br,bc);
         // setup pointer for conjugate transpose
         Tm* ptr = const_cast<Tm*>((*this)(bc,br).data());
         qt2(br,bc).setup_data(ptr);
      } // i
   }
   return qt2;
}

template <typename Tm>
linalg::matrix<Tm> stensor2<Tm>::to_matrix() const{
   int m = info.qrow.get_dimAll();
   int n = info.qcol.get_dimAll();
   linalg::matrix<Tm> mat(m,n);
   // assign block to proper place
   auto roff = info.qrow.get_offset();
   auto coff = info.qcol.get_offset();
   for(int br=0; br<info._rows; br++){
      int offr = roff[br];		 
      for(int bc=0; bc<info._cols; bc++){
         int offc = coff[bc];
	 const auto& blk = (*this)(br,bc);
         if(blk.size() == 0) continue;
         for(int ic=0; ic<blk.dim1; ic++){
     	    for(int ir=0; ir<blk.dim0; ir++){
               mat(offr+ir,offc+ic) = blk(ir,ic);
            } // ir
         } // ic
      } // bc
   } // br
   return mat;
}

// from dense matrix: assign block to proper place
template <typename Tm>
void stensor2<Tm>::from_matrix(const linalg::matrix<Tm>& mat){
   auto roff = info.qrow.get_offset();
   auto coff = info.qcol.get_offset();
   for(int br=0; br<info._rows; br++){
      int offr = roff[br];		 
      for(int bc=0; bc<info._cols; bc++){
         int offc = coff[bc];
         auto& blk = (*this)(br,bc);
         if(blk.size() == 0) continue;
         for(int ic=0; ic<blk.dim1; ic++){
   	    for(int ir=0; ir<blk.dim0; ir++){
               blk(ir,ic) = mat(offr+ir,offc+ic);
            } // ir
         } // ic
      } // bc
   } // br
}

template <typename Tm>
double stensor2<Tm>::check_identityMatrix(const double thresh_ortho, const bool debug) const{
   if(debug) std::cout << "stensor2::check_identityMatrix thresh_ortho=" << thresh_ortho << std::endl;
   double maxdiff = -1.0;
   for(int br=0; br<info._rows; br++){
      for(int bc=0; bc<info._cols; bc++){
         const auto& blk = (*this)(br,bc);
         if(blk.size() == 0) continue;
	 if(br != bc){
	    std::string msg = "error: not a block-diagonal matrix! br,bc=";
	    tools::exit(msg+std::to_string(br)+","+std::to_string(bc));
	 }
         auto qr = info.qrow.get_sym(br);
         int ndim = info.qrow.get_dim(br);
         double diff = linalg::normF(blk.to_matrix() - linalg::identity_matrix<Tm>(ndim));
	 maxdiff = std::max(diff,maxdiff);
	 if(debug || (!debug && diff > thresh_ortho)){ 
	    std::cout << " br=" << br << " qr=" << qr << " ndim=" << ndim 
		      << " |Sr-Id|_F=" << diff << std::endl;
	 }
         if(diff > thresh_ortho){
	    blk.print("diagonal block");
	    tools::exit("error: not an identity matrix!"); 
	 }
      } // bc
   } // br
   return maxdiff;
}

// ZL20200531: Permute the line of diagrams, while maintaining their directions
// 	       This does not change the tensor, but just permute order of index
template <typename Tm>
stensor2<Tm> stensor2<Tm>::T() const{
   //std::cout << "stensor2: T()" << std::endl;
   stensor2<Tm> qt2(info.sym, info.qcol, info.qrow, {info.dir[1], info.dir[0]});
   int br, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int addr = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(addr,br,bc);
      auto& blk = qt2(br,bc);
      // transpose
      const auto& blkt = (*this)(bc,br);
      for(int ic=0; ic<blk.dim1; ic++){
         for(int ir=0; ir<blk.dim0; ir++){
            blk(ir,ic) = blkt(ic,ir);
         } // ir
      } // ic
   } // i
   return qt2; 
}

// ZL20200531: This is used in taking Hermitian conjugate of operators.
// 	       If row/col is permuted while dir fixed, this effectively changes 
// 	       the direction of lines in diagrams
template <typename Tm>
stensor2<Tm> stensor2<Tm>::H() const{
   //std::cout << "stensor2: H()" << std::endl;
   // symmetry of operator get changed in consistency with line changes
   stensor2<Tm> qt2(-info.sym, info.qcol, info.qrow, info.dir);
   int br, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int addr = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(addr,br,bc);
      auto& blk = qt2(br,bc);
      // conjugate transpose
      const auto& blkh = (*this)(bc,br);
      for(int ic=0; ic<blk.dim1; ic++){
         for(int ir=0; ir<blk.dim0; ir++){
	    blk(ir,ic) = tools::conjugate(blkh(ic,ir));
	 } // ir
      } // ic
   } // i
   return qt2; 
}

// generate matrix representation for Kramers paired operators
// suppose row and col are KRS-adapted basis, then
//    <r|\bar{O}|c> = (K<r|\bar{O}|c>)*
//    		    = p{O} <\bar{r}|O|\bar{c}>*
// using \bar{\bar{O}} = p{O} O (p{O}: 'parity' of operator)
template <typename Tm>
stensor2<Tm> stensor2<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   // the symmetry is flipped
   stensor2<Tm> qt2(info.sym.flip(), info.qrow, info.qcol, info.dir);
   int br, bc;
   for(int i=0; i<qt2.info._nnzaddr.size(); i++){
      int addr = qt2.info._nnzaddr[i];
      qt2.info._addr_unpack(addr,br,bc);
      auto& blk = qt2(br,bc);
      // kramers 
      const auto& blkk = (*this)(br,bc);
      int pr = info.qrow.get_parity(br);
      int pc = info.qcol.get_parity(bc);
      auto mat = blkk.time_reversal(pr, pc);
      linalg::xaxpy(blk.size(), fpo, mat.data(), blk.data());
   } // i
   return qt2; 
}

} // ctns

#endif
