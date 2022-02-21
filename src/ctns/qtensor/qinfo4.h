#ifndef QINFO4_H
#define QINFO4_H

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
	    & _rows & _cols & _mids & _vers & _size 
	    & _nnzaddr & _qblocks;
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
      // --- derived --- 
      int _rows, _cols, _mids, _vers;
      size_t _size = 0;
      std::vector<int> _nnzaddr;
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
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
	 for(int bm=0; bm<qmid.size(); bm++){
	    int mdim = qmid.get_dim(bm);
	    for(int bv=0; bv<qver.size(); bv++){
	       int vdim = qver.get_dim(bv);
	       if(not _ifconserve(br,bc,bm,bv)) continue;
	       int addr = _addr(br,bc,bm,bv);
	       _nnzaddr.push_back(addr);
	       _qblocks[addr].setup_dims(rdim,cdim,mdim,vdim);
	       _size += rdim*cdim*mdim*vdim;
	    } // bv
	 } // bm
      } // bc
   } // br
}

template <typename Tm>
void qinfo4<Tm>::setup_data(Tm* data){
   size_t off = 0;
   for(int i=0; i<_nnzaddr.size(); i++){
      int addr = _nnzaddr[i];
      _qblocks[addr].setup_data(data+off);
      off += _qblocks[addr].size();
   }
}

template <typename Tm>
void qinfo4<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqinfo4: " << name << " sym=" << sym << std::endl;
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   qver.print("qver");
   // qblocks
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size() 
             << " nblocks=" << _qblocks.size() 
             << " size=" << _size << ":" 
             << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl; 
   int br, bc, bm, bv;
   for(int i=0; i<_nnzaddr.size(); i++){
      int idx = _nnzaddr[i];
      _addr_unpack(idx,br,bc,bm,bv);	   
      const auto& blk = _qblocks[idx];
      if(level >= 1){
         std::cout << " inz=" << i << " idx=" << idx 
         	   << " block[" << qrow.get_sym(br) << "," << qcol.get_sym(bc) << ","
                   << qmid.get_sym(bm) << "," << qver.get_sym(bv) << "]" 
                   << " dim0,dim1,dim2,dim3=(" << blk.dim0 << "," << blk.dim1 << "," 
                   << blk.dim2 << "," << blk.dim3 << ")" 
                   << std::endl; 
         if(level >= 2) blk.print("blk_"+std::to_string(idx));
      } // level>=1
   } // idx
}

} // ctns

#endif
