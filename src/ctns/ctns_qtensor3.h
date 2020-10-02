#ifndef CTNS_QTENSOR3_H
#define CTNS_QTENSOR3_H

#include <vector>
#include <string>
#include <map>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "ctns_qsym.h"
#include "ctns_qtensor2.h"

namespace ctns{

// rank-3 tensor: (mid,row,col)
template <typename Tm>
struct qtensor3{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & dir & sym & qmid & qrow & qcol 
	    & _mids & _rows & _cols & _qblocks;     
      }
      // conservation pattern determined by dir
      bool _ifconserve(const int bm, const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qmid.get_sym(bm) : -qmid.get_sym(bm);
	 qsum += dir[1] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[2] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum == qsym(0,0);
      }
      // address for storaging block data 
      int _addr(const int bm, const int br, const int bc) const{
         return bm*_rows*_cols + br*_cols + bc;
      }
      void _addr_unpack(const int idx, int& bm, int& br, int& bc) const{
	 int mr = idx/_cols;
	 bm = mr/_rows;	    
         br = mr%_rows;
 	 bc = idx%_cols;
      }
   public:
      // constructor
      qtensor3(){}
      qtensor3(const qsym& sym1,
	       const qsym_space& qmid1,
	       const qsym_space& qrow1, 
	       const qsym_space& qcol1,
	       const std::vector<bool> dir1={1,0,1}): 
	sym(sym1), qmid(qmid1), qrow(qrow1), qcol(qcol1), dir(dir1)
      {
         _mids = qmid.size();
	 _rows = qrow.size();
	 _cols = qcol.size();
	 _qblocks.resize(_mids*_rows*_cols); 
	 for(int bm=0; bm<_mids; bm++){
            for(int br=0; br<_rows; br++){
	       for(int bc=0; bc<_cols; bc++){
		  if(not _ifconserve(bm,br,bc)) continue;
		  int mdim = qmid.get_dim(bm);
		  int rdim = qrow.get_dim(br);
		  int cdim = qcol.get_dim(bc);
		  int addr = _addr(bm,br,bc);
		  _qblocks[addr].resize(mdim);
		  for(int im=0; im<mdim; im++){
		     _qblocks[addr][im].resize(rdim,cdim);
		  }
	       } // bc
	    } // br
	 } // bm
      }
      // helpers
      int mids() const{ return _mids; }
      int rows() const{ return _rows; }
      int cols() const{ return _cols; }
      // access
      std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int br, const int bc){
         return _qblocks[_addr(bm,br,bc)];
      }
      const std::vector<linalg::matrix<Tm>>& operator ()(const int bm, const int br, const int bc) const{
         return _qblocks[_addr(bm,br,bc)];
      }
      // fix middle index
      qtensor2<Tm> fix_mid(const std::pair<int,int> mdx) const{
	 int bm = mdx.first, im = mdx.second;   
	 auto symIn = dir[0] ? sym-qmid.get_sym(bm) : sym+qmid.get_sym(bm);
         qtensor2<Tm> qt2(symIn, qrow, qcol, {dir[1],dir[2]});
       	 for(int br=0; br<_rows; br++){
	    for(int bc=0; bc<_cols; bc++){
	       if(not _ifconserve(bm,br,bc)) continue;
	       int addr = _addr(bm,br,bc);
	       int mdim = qmid.get_dim(bm);
	       assert(im < mdim);
	       qt2(br,bc) = _qblocks[addr][im];
	    } // bc
	 } // br
	 return qt2;
      }
      // print
      void print(const std::string name, const int level=0) const{
	 std::cout << "\nqtensor3: " << name << " sym=" << sym;
	 std::cout << " dir=";
         for(auto b : dir) std::cout << b << " ";
	 std::cout << std::endl;
         qmid.print("qmid");
         qrow.print("qrow");
         qcol.print("qcol");
         // qblocks
	 std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
         int nnz = 0;
         for(int idx=0; idx<_qblocks.size(); idx++){
            auto& blk = _qblocks[idx];
	    if(blk.size() > 0){
               nnz++;
               if(level >= 1){
	          int bm,br,bc;
	          _addr_unpack(idx,bm,br,bc);
                  std::cout << "idx=" << idx 
           	       << " block[" << qmid.get_sym(bm) << "," 
		       << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "]" 
                       << " size=" << blk.size() 
                       << " rows,cols=(" << blk[0].rows() << "," << blk[0].cols() << ")" 
                       << std::endl; 
                  if(level >= 2){
                     for(int im=0; im<blk.size(); im++){
                        blk[im].print("mat"+std::to_string(im));
                     }
                  } // level=2
      	       } // level>=1
            }
         } // idx
	 std::cout << "total no. of nonzero blocks=" << nnz << std::endl;
      }
      // deal with fermionic sign in fermionic direct product
      qtensor3<Tm> mid_signed(const double fac=1.0) const{
	 qtensor3<Tm> qt3 = *this;
	 for(int idx=0; idx<qt3._qblocks.size(); idx++){
	    auto& blk = qt3._qblocks[idx];
	    if(blk.size() > 0){
	       int bm,br,bc;
	       _addr_unpack(idx,bm,br,bc);
	       double fac2 = (qmid.get_parity(bm)==0)? fac : -fac;
	       for(int im=0; im<blk.size(); im++){
	          blk[im] *= fac2;
	       }
	    }
	 }
	 return qt3;
      }
      qtensor3<Tm> row_signed(const double fac=1.0) const{
 	 qtensor3<Tm> qt3 = *this;
	 for(int idx=0; idx<qt3._qblocks.size(); idx++){
	    auto& blk = qt3._qblocks[idx];
	    if(blk.size() > 0){
	       int bm,br,bc;
	       _addr_unpack(idx,bm,br,bc);
	       double fac2 = (qrow.get_parity(br)==0)? fac : -fac;
	       for(int im=0; im<blk.size(); im++){
	          blk[im] *= fac2;
	       }
	    }
	 }
	 return qt3;
      }
      // simple algrithmic operations
      qtensor3<Tm>& operator +=(const qtensor3<Tm>& qt){
         assert(dir == qt.dir); // direction must be the same
         assert(sym == qt.sym); // symmetry blocking must be the same
         for(int i=0; i<_qblocks.size(); i++){
            auto& blk = _qblocks[i];
            assert(blk.size() == qt._qblocks[i].size());
	    if(blk.size() > 0){
	       for(int m=0; m<blk.size(); m++){
		  blk[m] += qt._qblocks[i][m];
	       } // m
	    }
         }
         return *this;
      }
      qtensor3<Tm>& operator -=(const qtensor3<Tm>& qt){
         assert(dir == qt.dir); // direction must be the same
         assert(sym == qt.sym); // symmetry blocking must be the same
         for(int i=0; i<_qblocks.size(); i++){
            auto& blk = _qblocks[i];
            assert(blk.size() == qt._qblocks[i].size());
	    if(blk.size() > 0){
	       for(int m=0; m<blk.size(); m++){
		  blk[m] -= qt._qblocks[i][m];
	       } // m
	    }
         }
         return *this;
      }
      qtensor3<Tm>& operator *=(const Tm fac){
         for(auto& blk : _qblocks){
            if(blk.size() > 0){ 
      	       for(int m=0; m<blk.size(); m++){
      	          blk[m] *= fac;
      	       } // m
            }
         }
         return *this;
      }
      friend qtensor3<Tm> operator +(const qtensor3<Tm>& qta, const qtensor3<Tm>& qtb){
         qtensor3<Tm> qt3 = qta;
         qt3 += qtb;
         return qt3;
      }
      friend qtensor3<Tm> operator -(const qtensor3<Tm>& qta, const qtensor3<Tm>& qtb){
         qtensor3<Tm> qt3 = qta;
         qt3 -= qtb;
         return qt3;
      }
      friend qtensor3<Tm> operator *(const double fac, const qtensor3<Tm>& qt){
         qtensor3 qt3 = qt; 
         qt3 *= fac;
         return qt3;
      }
      friend qtensor3<Tm> operator *(const qtensor3<Tm>& qt, const double fac){
         return fac*qt;
      }
      // for Davidson algorithm
      double normF() const{
         double sum = 0.0;
         for(const auto& blk : _qblocks){
            if(blk.size() > 0){
      	       for(int m=0; m<blk.size(); m++){
                  sum += std::pow(linalg::normF(blk[m]),2);
               }
            }
         }
         return std::sqrt(sum);
      }
      int get_dim() const{
         int dim = 0;
         for(const auto& blk : _qblocks){
            if(blk.size() > 0){
               dim += blk.size()*blk[0].size(); // A[l,c,r] = A[c](l,r)
            }
         }
         return dim;
      }
      //void from_array(const double* array);
      //void to_array(double* array) const;
      
/*
      // extract real & imag parts
      matrix<double> real() const{
	 matrix<double> matr(_rows,_cols);
	 std::transform(_data, _data+_size, matr._data,
			[](const Tm& x){ return std::real(x); });
	 return matr;
      }
      matrix<double> imag() const{
	 matrix<double> mati(_rows,_cols);
	 std::transform(_data, _data+_size, mati._data,
			[](const Tm& x){ return std::imag(x); });
	 return mati;
      }
*/

   public:
      std::vector<bool> dir = {1,0,1}; // =0,in; =1,out; {mid,row,col}
      				       // {1,0,1} - RCF (default)
      				       // {1,1,0} - LCF
				       // {0,1,1} - CCF (for internal upward node)
				       // {1,1,1} - WF
      qsym sym; // in
      qsym_space qmid, qrow, qcol; 
   private:  
      int _mids, _rows, _cols; 
      std::vector<std::vector<linalg::matrix<Tm>>> _qblocks;
};

} // ctns

#endif
