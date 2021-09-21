#ifndef TENSOR_QINFO4_H
#define TENSOR_QINFO4_H

#include "../../core/serialization.h"
#include "../../core/tools.h"

namespace ctns{

template <typename Tm>
struct qinfo4{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & qmid & qver 
	    & _rows & _cols & _mids & _vers & _size & _qblocks;
      }
   public:
      // --- GENERAL FUNCTIONS ---
      // conservation: dir={1,1,1,1} 
      bool _ifconserve(const int br, const int bc, const int bm, const int bv) const{
	 return sym == qrow.get_sym(br) + qcol.get_sym(bc) + qmid.get_sym(bm) + qver.get_sym(bv);
      }
      // address for storaging block data
      int _addr(const int br, const int bc, const int bm, const int bv) const{
	 return ((br*_cols + bc)*_mids + bm)*_vers + bv;
      }
      void _addr_unpack(const int idx4, int& br, int& bc, int& bm, int& bv) const{
         bv = idx4%_vers;
         int idx3 = idx4/_vers;
	 bm = idx3%_mids;
	 int idx2 = idx3/_mids;
	 bc = idx2%_cols;
	 br = idx2/_cols; 
      }
      // initialization
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
	        const qbond& _qmid, const qbond& _qver);
      void setup_data(Tm* data);
      // print
      void print(const std::string name, const int level=0) const;
      // --- SPECIFIC FUNCTIONS ---
      bool operator ==(const qinfo4& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
		 && qmid==info.qmid && qver==info.qver;
      }
   public:
      qsym sym;
      qbond qrow, qcol, qmid, qver;
   public:
      int _rows, _cols, _mids, _vers;
      size_t _size = 0;
      std::vector<dtensor4<Tm>> _qblocks;
};

template <typename Tm>
void qinfo4<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
		      const qbond& _qmid, const qbond& _qver){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   qmid = _qmid;
   qver = _qver;
   _rows = qrow.size();
   _cols = qcol.size();
   _mids = qmid.size();
   _vers = qver.size();
   _qblocks.resize(_rows*_cols*_mids*_vers);
   int br, bc, bm, bv;
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
	 for(int bm=0; bm<qmid.size(); bm++){
	    int mdim = qmid.get_dim(bm);
	    for(int bv=0; bv<qver.size(); bv++){
	       int vdim = qver.get_dim(bv);
	       if(not _ifconserve(br,bc,bm,bv)) continue;
	       _qblocks[_addr(br,bc,bm,bv)].setup_dims(rdim,cdim,mdim,vdim,_size);
	       _size += rdim*cdim*mdim*vdim;
	    } // bv
	 } // bm
      } // bc
   } // br
}

template <typename Tm>
void qinfo4<Tm>::setup_data(Tm* data){
   for(auto& blk : _qblocks){
      if(blk.size() == 0) continue;
      blk.setup_data(data);
   } // blk
}

template <typename Tm>
void qinfo4<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqinfo2: " << name << " sym=" << sym;
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   qver.print("qver");
   // qblocks
   std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
   int nnz = 0, br, bc, bm, bv;
   for(int idx=0; idx<_qblocks.size(); idx++){  
      _addr_unpack(idx,br,bc,bm,bv);	   
      const auto& blk = _qblocks[idx];
      if(blk.size() > 0){
         nnz++;
         if(level >= 1){
            std::cout << "idx=" << idx 
     	    	      << " block[" << qrow.get_sym(br) << "," << qcol.get_sym(bc) << ","
		      << qmid.get_sym(bm) << "," << qver.get_sym(bv) << "]" 
                      << " dim0,dim1,dim2,dim3=(" << blk.dim0 << "," << blk.dim1 << "," 
		      << blk.dim2 << "," << blk.dim3 << ")" 
                      << std::endl; 
            //if(level >= 2) blk.print("blk_"+std::to_string(idx));
	 } // level>=1
      }
   } // idx
   std::cout << "total no. of nonzero blocks=" << nnz << std::endl;
   std::cout << "total size=" << _size << " sizeMB=" << tools::sizeMB<Tm>(_size) << std::endl; 
}

} // ctns

#endif
