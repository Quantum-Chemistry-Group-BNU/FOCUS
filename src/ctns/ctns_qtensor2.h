#ifndef CTNS_QTENSOR2_H
#define CTNS_QTENSOR2_H

#include <vector>
#include <string>
#include <map>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "ctns_qsym.h"
#include "ctns_qtensor_contract.h"

namespace ctns{

// rank-2 tensor: (row,col)
template <typename Tm>
struct qtensor2{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dir & sym & qrow & qcol 
	    & _rows & _cols & _qblocks;     
      }
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[1] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum == qsym(0,0);
      }
      // address for storaging block data 
      int _addr(const int br, const int bc) const{
         return br*_cols + bc;
      }
   public:
      // constructor
      qtensor2(){}
      qtensor2(const qsym& sym1,
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1,
	       const std::vector<bool> dir1={1,0}): 
	sym(sym1), qrow(qrow1), qcol(qcol1), dir(dir1)
      {
	 _rows = qrow.size();
	 _cols = qcol.size();
	 _qblocks.resize(_rows*_cols); 
         for(int br=0; br<_rows; br++){
	    for(int bc=0; bc<_cols; bc++){
	       if(not _ifconserve(br,bc)) continue;
	       int rdim = qrow.get_dim(br);
	       int cdim = qcol.get_dim(bc);
	       int addr = _addr(br,bc);
	       _qblocks[addr].resize(rdim,cdim);
	    } // bc
	 } // br
      }
      // helpers
      int rows() const{ return _rows; }
      int cols() const{ return _cols; }
      // access
      linalg::matrix<Tm>& operator ()(const int br, const int bc){
         return _qblocks[_addr(br,bc)];
      }
      const linalg::matrix<Tm>& operator ()(const int br, const int bc) const{
         return _qblocks[_addr(br,bc)];
      }
      // print
      void print(const std::string name, const int level=0) const{
	 std::cout << "\nqtensor2: " << name << " sym=" << sym;
	 std::cout << " dir=";
         for(auto b : dir) std::cout << b << " ";
	 std::cout << std::endl;
         qrow.print("qrow");
         qcol.print("qcol");
         // qblocks
	 std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
         int nnz = 0;
         for(int idx=0; idx<_qblocks.size(); idx++){
 	    int bc = idx%_cols;
	    int br = idx/_cols;
            auto& blk = _qblocks[idx];
	    if(blk.size() > 0){
               nnz++;
               if(level >= 1){
                  std::cout << "idx=" << idx 
           	       << " block[" << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "]" 
                       << " rows,cols=(" << blk.rows() << "," << blk.cols() << ")" 
                       << std::endl; 
                  if(level >= 2) blk.print("blk");
      	       } // level>=1
            }
         } // idx
	 std::cout << "total no. of nonzero blocks=" << nnz << std::endl;
      }
      // check whether <l|o|r> is a faithful rep for o=I
      double check_identity(const double thresh_ortho, const bool debug) const{
         if(debug) std::cout << "qtensor2::check_identity thresh_ortho=" 
		  	     << thresh_ortho << std::endl;
         double maxdiff = -1.0;
         for(int br=0; br<_rows; br++){
            for(int bc=0; bc<_cols; bc++){
	       const auto& blk = _qblocks[_addr(br,bc)];
	       if(blk.size() > 0){
      	          if(br != bc){
      	             std::cout << "error: not a block-diagonal matrix! br,bc="
			       << br << "," << bc << std::endl;
      	             exit(1);
      	          }
	          auto qr = qrow.get_sym(br);
                  int ndim = qrow.get_dim(br);
                  double diff = linalg::normF(blk - linalg::identity_matrix<Tm>(ndim));
      	          maxdiff = std::max(diff,maxdiff);
      	          if(debug || diff > thresh_ortho){
	             std::cout << "qsym=" << qr << " ndim=" << ndim 
                    	       << " |Sr-Id|_F=" << diff << std::endl;
      	          }
                  if(diff > thresh_ortho){
      	             std::cout << "error: not an identity matrix at qsym! thresh_ortho=" 
			       << thresh_ortho << std::endl;
      	             blk.print("diagonal block");
		     exit(1);
      	          }
	       } // blk
            } // bc
	 } // br
         return maxdiff;
      }
      // convert to matrix class
      linalg::matrix<Tm> to_matrix() const{
	 int m = qrow.get_dimAll();
	 int n = qcol.get_dimAll();
	 linalg::matrix<Tm> mat(m,n);
	 // assign block to proper place
	 auto roff = qrow.get_offset();
	 auto coff = qcol.get_offset();
	 for(int br=0; br<_rows; br++){
 	    int offr = roff[br];		 
	    for(int bc=0; bc<_cols; bc++){
	       auto& blk = _qblocks[_addr(br,bc)];
	       if(blk.size() == 0) continue;
	       int offc = coff[bc];
	       for(int ic=0; ic<qcol.get_dim(bc); ic++){
	 	  for(int ir=0; ir<qrow.get_dim(br); ir++){
		     mat(offr+ir,offc+ic) = blk(ir,ic);
		  } // ir
	       } // ic
	    } // bc
	 } // br
	 return mat;
      }
      // xgemm
      qtensor2<Tm> dot(const qtensor2<Tm>& qt) const{
         return contract_qt2_qt2(*this, qt);
      }
      // ZL20200531: permute the line of diagrams, while maintaining their directions
      // 	     This does not change the tensor, but just permute order of index
      qtensor2<Tm> T() const{
         qtensor2<Tm> qt2(sym, qcol, qrow, {dir[1],dir[0]});
         for(int br=0; br<qt2.rows(); br++){
	    for(int bc=0; bc<qt2.cols(); bc++){
	       auto& blk = qt2(br,bc);
	       if(blk.size() == 0) continue;
	       blk = _qblocks[_addr(bc,br)].T(); 
	    }
	 }
	 return qt2; 
      }
      // ZL20200531: if row/col is permuted while dir fixed, 
      // 	     effectively change the direction of lines in diagrams
      //	     This is used in taking Hermitian conjugate of operators
      qtensor2<Tm> H() const{
	 // symmetry of operator get changed in consistency with line changes
         qtensor2<Tm> qt2(-sym, qcol, qrow, dir);
         for(int br=0; br<qt2.rows(); br++){
	    for(int bc=0; bc<qt2.cols(); bc++){
	       auto& blk = qt2(br,bc);
	       if(blk.size() == 0) continue;
	       blk = _qblocks[_addr(bc,br)].H(); 
	    }
	 }
	 return qt2; 
      }
   public:
      std::vector<bool> dir = {1,0}; // {out,int} by usual convention for operators in diagrams
      qsym sym; // <row|op[in]|col>
      qsym_space qrow, qcol; 
   private:  
      int _rows, _cols; 
      std::vector<linalg::matrix<Tm>> _qblocks;
};

} // ctns

#endif
