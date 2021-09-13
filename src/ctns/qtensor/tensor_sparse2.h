#ifndef TENSOR_SPARSE2_H
#define TENSOR_SPARSE2_H

#include "../../core/serialization.h"
#include "../../core/matrix.h"
#include "../../core/linalg.h"

namespace ctns{

template <typename Tm>
struct stensor2{
   public:
      // --- GENERAL FUNCTIONS ---
      // constructors
      stensor2(): _data(nullptr) {};
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
	        const std::vector<bool> _dir={1,0}){
         info.init(_sym, _qrow, _qcol, _dir);
         _data = new Tm[info._size];
	 info.setup_data(_data);
      }
      stensor2(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
	       const std::vector<bool> _dir={1,0}){
	 this->init(_sym, _qrow, _qcol, _dir);     
      }
      // desctructors
      ~stensor2(){ delete[] _data; }
      stensor2(const stensor2& st) = delete;
      stensor2& operator =(const stensor2& st) = delete;
      /*
      // copy constructor
      stensor2(const stensor2& st){
	 //std::cout << "stensor2: copy constructor" << std::endl;     
         info = st.info;
	 _data = new Tm[info._size];
	 std::copy_n(st._data, info._size, _data);
	 info.setup_data(_data);
      }
      // copy assignment
      stensor2& operator =(const stensor2& st){
	 //std::cout << "stensor2: copy assignment" << std::endl;     
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
      stensor2(stensor2&& st){
	 //std::cout << "stensor2: move constructor" << std::endl;     
         info = std::move(st.info);
         _data = st._data;
	 st._data = nullptr;
      }
      // move assignment
      stensor2& operator =(stensor2&& st){
	 //std::cout << "stensor2: move assignment" << std::endl;     
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
      int row_dim(const int br) const{ return info.qrow.get_dim(br); }
      int col_dim(const int bc) const{ return info.qcol.get_dim(bc); } 
      int row_dimAll() const{ return info.qrow.get_dimAll(); }
      int col_dimAll() const{ return info.qcol.get_dimAll(); } 
      size_t size() const{ return info._size; }
      // print
      void print(const std::string name, const int level=0) const{ info.print(name,level); }
      // access
      const dtensor2<Tm> operator()(const int br, const int bc) const{ 
	 return info._qblocks[info._addr(br,bc)]; 
      }
      dtensor2<Tm>& operator()(const int br, const int bc){ 
	 return info._qblocks[info._addr(br,bc)]; 
      }
      // --- SPECIFIC FUNCTIONS ---
      // convert to matrix class
      linalg::matrix<Tm> to_matrix() const;
      // from dense matrix: assign block to proper place
      void from_matrix(const linalg::matrix<Tm>& mat);
      // check whether <l|o|r> is a faithful rep for o=I
      double check_identityMatrix(const double thresh_ortho, const bool debug) const;
   public:
      qinfo2<Tm> info;
   private:   
      Tm* _data;
};

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

} // ctns

#endif
