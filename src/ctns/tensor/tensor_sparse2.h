#ifndef TENSOR_SPARSE2_H
#define TENSOR_SPARSE2_H

#include "../../core/serialization.h"

namespace ctns{

template <typename Tm>
struct stensor2{
   public:
      // constructors
      stensor2(){};
      stensor2(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const std::vector<bool> _dir={1,0}){
         info2.init(_sym, _qrow, _qcol, _dir);
         data = new Tm[info2._size];
	 info2.setup_qblocks(data);
      }
      // desctructors
      ~stensor2(){ delete[] data; }
      // helpers
      int rows() const{ return info2._rows; }
      int cols() const{ return info2._cols; }
      // access
      dtensor2<Tm>& operator ()(const int br, const int bc){ return info2._qblocks[info2._addr(br,bc)]; }
      const dtensor2<Tm>& operator ()(const int br, const int bc) const{ return info2._qblocks[info2._addr(br,bc)]; }
      // convert to matrix class
      linalg::matrix<Tm> to_matrix() const;
      // from dense matrix: assign block to proper place
      void from_matrix(const linalg::matrix<Tm>& mat);
   public:
      qinfo2<Tm> info2;
      Tm* data;
};

template <typename Tm>
linalg::matrix<Tm> stensor2<Tm>::to_matrix() const{
   int m = info2.qrow.get_dimAll();
   int n = info2.qcol.get_dimAll();
   linalg::matrix<Tm> mat(m,n);
   // assign block to proper place
   auto roff = info2.qrow.get_offset();
   auto coff = info2.qcol.get_offset();
   for(int br=0; br<this->rows(); br++){
      int offr = roff[br];		 
      for(int bc=0; bc<this->cols(); bc++){
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
   auto roff = info2.qrow.get_offset();
   auto coff = info2.qcol.get_offset();
   for(int br=0; br<this->rows(); br++){
      int offr = roff[br];		 
      for(int bc=0; bc<this->cols(); bc++){
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

} // ctns

#endif
