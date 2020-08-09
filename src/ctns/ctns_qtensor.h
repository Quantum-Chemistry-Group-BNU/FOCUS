#ifndef CTNS_QTENSOR_H
#define CTNS_QTENSOR_H

#include <vector>
#include <string>
#include <map>
#include "../core/serialization.h"
#include "../core/matrix.h"
#include "ctns_qsym.h"

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
      bool _ifconserve(const int bm, const int br, const int bc){
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qmid.get_sym(bm) : -qmid.get_sym(bm);
	 qsum += dir[1] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[2] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum == qsym(0,0);
      }
      // address for storaging block data 
      int _addr(const int bm, const int br, const int bc){
         return bm*_rows*_cols + br*_cols + bc;
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
 	    int bc = idx%_cols;
	    int mr = idx/_cols;
            int br = mr%_rows;
	    int bm = mr/_rows;	    
            auto& blk = _qblocks[idx];
	    if(blk.size() > 0){
               nnz++;
               if(level >= 1){
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
      // assignment
      std::vector<linalg::matrix<Tm>>& operator ()(const int im, 
		      		 		   const int ir,
						   const int ic){
         return _qblocks[_addr(im,ir,ic)];
      }
   public:
      std::vector<bool> dir = {1,0,1}; // =0,in; =1,out; {mid,row,col}
      				       // {1,0,1} - RCF (default)
      				       // {1,1,0} - LCF
				       // {0,1,1} - CCF (for internal upward node)
				       // {1,1,1} - WF
      qsym sym; // in
      qsym_space qmid; 
      qsym_space qrow; 
      qsym_space qcol; 
   private:  
      int _mids, _rows, _cols; 
      std::vector<std::vector<linalg::matrix<Tm>>> _qblocks;
};

} // ctns

#endif
