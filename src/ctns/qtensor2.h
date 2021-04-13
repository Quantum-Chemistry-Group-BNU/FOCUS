#ifndef QTENSOR2_H
#define QTENSOR2_H

#include <vector>
#include <string>
#include <map>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "../core/linalg.h"
#include "ctns_qsym.h"
#include "ctns_kramers.h"
#include "qtensor_contract.h"

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
      inline bool _ifconserve(const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[1] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum == qsym(0,0);
      }
      // address for storaging block data 
      inline int _addr(const int br, const int bc) const{ return br*_cols + bc; }
   public:
      // constructor
      qtensor2(){}
      qtensor2(const qsym& sym1, const qbond& qrow1, const qbond& qcol1, const std::vector<bool> dir1={1,0}); 
      void init(const qsym& sym1, const qbond& qrow1, const qbond& qcol1, const std::vector<bool> dir1={1,0});
      // helpers
      inline int rows() const{ return _rows; }
      inline int cols() const{ return _cols; }
      // access
      linalg::matrix<Tm>& operator ()(const int br, const int bc){ return _qblocks[_addr(br,bc)]; }
      const linalg::matrix<Tm>& operator ()(const int br, const int bc) const{ return _qblocks[_addr(br,bc)]; }
      // print
      void print(const std::string name, const int level=0) const;
      // check whether <l|o|r> is a faithful rep for o=I
      double check_identityMatrix(const double thresh_ortho, const bool debug) const;
      // convert to matrix class
      linalg::matrix<Tm> to_matrix() const;
      // from dense matrix: assign block to proper place
      void from_matrix(const linalg::matrix<Tm>& mat);
      // ZL20200531: permute the line of diagrams, while maintaining their directions
      // 	     This does not change the tensor, but just permute order of index
      qtensor2<Tm> T() const;
      // ZL20200531: if row/col is permuted while dir fixed, 
      // 	     effectively change the direction of lines in diagrams
      //	     This is used in taking Hermitian conjugate of operators
      qtensor2<Tm> H() const;
      // ZL20210401: generate matrix representation for Kramers paired operators
      qtensor2<Tm> K(const int nbar) const;
      // simple algorithmic operations
      qtensor2<Tm> operator -() const;
      qtensor2<Tm>& operator +=(const qtensor2<Tm>& qt);
      qtensor2<Tm>& operator -=(const qtensor2<Tm>& qt);
      qtensor2<Tm>& operator *=(const Tm fac);
      friend qtensor2<Tm> operator +(const qtensor2<Tm>& qta, const qtensor2<Tm>& qtb){
         qtensor2<Tm> qt2 = qta;
         qt2 += qtb;
         return qt2;
      }
      friend qtensor2<Tm> operator -(const qtensor2<Tm>& qta, const qtensor2<Tm>& qtb){
         qtensor2<Tm> qt2 = qta;
         qt2 -= qtb;
         return qt2;
      }
      friend qtensor2<Tm> operator *(const Tm fac, const qtensor2<Tm>& qt){
         qtensor2<Tm> qt2 = qt;
         qt2 *= fac;
         return qt2;
      }
      friend qtensor2<Tm> operator *(const qtensor2<Tm>& qt, const Tm fac){
         return fac*qt;
      }
      // algebra
      double normF() const;
      qtensor2<Tm> dot(const qtensor2<Tm>& qt) const;
      // for Davidson algorithm
      int get_dim() const;
      void random();
   public:
      std::vector<bool> dir = {1,0}; // {out,int} by usual convention for operators in diagrams
      qsym sym; // <row|op[in]|col>
      qbond qrow, qcol; 
   private:  
      int _rows, _cols; 
      std::vector<linalg::matrix<Tm>> _qblocks;
};

template <typename Tm>
void qtensor2<Tm>::init(const qsym& sym1, const qbond& qrow1, const qbond& qcol1,
	 	        const std::vector<bool> dir1){
   sym = sym1;
   qrow = qrow1;
   qcol = qcol1;
   dir = dir1;
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

template <typename Tm>
qtensor2<Tm>::qtensor2(const qsym& sym1, const qbond& qrow1, const qbond& qcol1,
	    	       const std::vector<bool> dir1){
   this->init(sym1, qrow1, qcol1, dir1);
}

template <typename Tm>
void qtensor2<Tm>::print(const std::string name, const int level) const{
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

template <typename Tm>
double qtensor2<Tm>::check_identityMatrix(const double thresh_ortho, const bool debug) const{
   if(debug) std::cout << "qtensor2::check_identityMatrix thresh_ortho=" << thresh_ortho << std::endl;
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
               std::cout << "qsym=" << qr << " ndim=" << ndim << " |Sr-Id|_F=" << diff << std::endl;
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

template <typename Tm>
linalg::matrix<Tm> qtensor2<Tm>::to_matrix() const{
   int m = qrow.get_dimAll();
   int n = qcol.get_dimAll();
   linalg::matrix<Tm> mat(m,n);
   // assign block to proper place
   auto roff = qrow.get_offset();
   auto coff = qcol.get_offset();
   for(int br=0; br<_rows; br++){
      int offr = roff[br];		 
      for(int bc=0; bc<_cols; bc++){
         const auto& blk = _qblocks[_addr(br,bc)];
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

// from dense matrix: assign block to proper place
template <typename Tm>
void qtensor2<Tm>::from_matrix(const linalg::matrix<Tm>& mat){
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
               blk(ir,ic) = mat(offr+ir,offc+ic);
            } // ir
         } // ic
      } // bc
   } // br
}

// ZL20200531: permute the line of diagrams, while maintaining their directions
// 	     This does not change the tensor, but just permute order of index
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::T() const{
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
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::H() const{
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

// generate matrix representation for Kramers paired operators
// suppose row and col are KRS-adapted basis, then
//    <r|\bar{O}|c> = (K<r|\bar{O}|c>)*
//    		    = p{O} <\bar{r}|O|\bar{c}>*
// using \bar{\bar{O}} = p{O} O (p{O}: 'parity' of operator)
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::K(const int nbar) const{
   const double fpo = (nbar%2==0)? 1.0 : -1.0;
   qtensor2<Tm> qt2(sym.flip(), qrow, qcol, dir); // the symmetry is flipped
   for(int br=0; br<qt2.rows(); br++){
      for(int bc=0; bc<qt2.cols(); bc++){
         auto& blk = qt2(br,bc);
         if(blk.size() == 0) continue;
	 const auto& blk1 = _qblocks[_addr(br,bc)];
	 int pr = qrow.get_parity(br);
	 int pc = qcol.get_parity(bc);
	 blk = fpo*time_reversal(blk1, pr, pc); 
      } // bc
   } // br
   return qt2;
}

// simple algorithmic operations
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::operator -() const{
   qtensor2<Tm> qt2 = *this;
   for(auto& blk : qt2._qblocks){
      if(blk.size() > 0) blk *= -1;
   }
   return qt2;
}

template <typename Tm>
qtensor2<Tm>& qtensor2<Tm>::operator +=(const qtensor2<Tm>& qt){
   assert(dir == qt.dir);
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(int i=0; i<_qblocks.size(); i++){
      auto& blk = _qblocks[i];
      assert(blk.size() == qt._qblocks[i].size());
      if(blk.size() > 0) blk += qt._qblocks[i];
   }
   return *this;
}

template <typename Tm>
qtensor2<Tm>& qtensor2<Tm>::operator -=(const qtensor2<Tm>& qt){
   assert(dir == qt.dir);
   assert(sym == qt.sym); // symmetry blocking must be the same
   for(int i=0; i<_qblocks.size(); i++){
      auto& blk = _qblocks[i];
      assert(blk.size() == qt._qblocks[i].size());
      if(blk.size() > 0) blk -= qt._qblocks[i];
   }
   return *this;
}

template <typename Tm>
qtensor2<Tm>& qtensor2<Tm>::operator *=(const Tm fac){
   for(auto& blk : _qblocks){
      if(blk.size() > 0) blk *= fac;
   }
   return *this;
}

// algebra
template <typename Tm>
double qtensor2<Tm>::normF() const{
   double sum = 0.0;
   for(const auto& blk : _qblocks){
      if(blk.size() > 0) sum += std::pow(linalg::normF(blk),2);
   }
   return std::sqrt(sum);
}

// xgemm
template <typename Tm>
qtensor2<Tm> qtensor2<Tm>::dot(const qtensor2<Tm>& qt) const{
   return contract_qt2_qt2(*this, qt);
}

// get_dim 
template <typename Tm> 
int qtensor2<Tm>::get_dim() const{
   int dim = 0;
   for(const auto& blk : _qblocks){
      if(blk.size() > 0) dim += blk.size();
   }
   return dim;
}

// random tensor
template<typename Tm>
void qtensor2<Tm>::random(){
   for(auto& blk : _qblocks){
      if(blk.size() > 0){
	 int rdim = blk.rows();
	 int cdim = blk.cols();
	 blk = linalg::random_matrix<Tm>(rdim, cdim);
      }
   }
}

} // ctns

#endif
