#ifndef QINFO3_H
#define QINFO3_H

#include "../../core/serialization.h"
#include "../../core/tools.h"

namespace ctns{

const std::vector<bool> dir_RCF = {0,1,1}; // RCF (default)
const std::vector<bool> dir_LCF = {1,0,1}; // LCF
const std::vector<bool> dir_CCF = {1,1,0}; // CCF (for internal upward node)
const std::vector<bool> dir_WF3 = {1,1,1}; // WF
extern const std::vector<bool> dir_RCF;
extern const std::vector<bool> dir_LCF;
extern const std::vector<bool> dir_CCF;
extern const std::vector<bool> dir_WF3;

template <typename Tm>
struct qinfo3{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & qmid & dir
	    & _rows & _cols & _mids & _size
	    & _nnzaddr & _qblocks;
      }
   public:
      // --- GENERAL FUNCTIONS ---
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc, const int bm) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[1] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 qsum += dir[2] ? qmid.get_sym(bm) : -qmid.get_sym(bm);
	 return qsum.is_zero();
      }
      // address for storaging block data
      int _addr(const int br, const int bc, const int bm) const{
	 return (br*_cols + bc)*_mids + bm;
      }
      void _addr_unpack(const int idx3, int& br, int& bc, int& bm) const{
	 bm = idx3%_mids;
	 int idx2 = idx3/_mids;
	 bc = idx2%_cols;
	 br = idx2/_cols; 
      }
      // initialization
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
		const std::vector<bool> _dir=dir_RCF);
      void setup_data(Tm* data);
      // print
      void print(const std::string name, const int level=0) const;
      // --- SPECIFIC FUNCTIONS ---
      bool operator ==(const qinfo3& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
		 && qmid==info.qmid && dir==info.dir;
      }
   public:
      qsym sym;
      qbond qrow, qcol, qmid;
      std::vector<bool> dir;
      // --- derived --- 
      int _rows, _cols, _mids;
      size_t _size = 0;
      std::vector<int> _nnzaddr;
      std::vector<dtensor3<Tm>> _qblocks;
};

template <typename Tm>
void qinfo3<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const qbond& _qmid,
	 	      const std::vector<bool> _dir){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   qmid = _qmid;
   dir = _dir;
   _rows = qrow.size();
   _cols = qcol.size();
   _mids = qmid.size();
   _qblocks.resize(_rows*_cols*_mids);
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
	 for(int bm=0; bm<qmid.size(); bm++){
	    int mdim = qmid.get_dim(bm);
	    if(not _ifconserve(br,bc,bm)) continue;
            int addr = _addr(br,bc,bm);
	    _nnzaddr.push_back(addr);
	    _qblocks[addr].setup_dims(rdim,cdim,mdim);
	    _size += rdim*cdim*mdim;
	 } // bm 
      } // bc
   } // br
}

template <typename Tm>
void qinfo3<Tm>::setup_data(Tm* data){
   size_t off = 0;
   for(int i=0; i<_nnzaddr.size(); i++){
      int addr = _nnzaddr[i];
      off += _qblocks[addr].size();
      _qblocks[addr].setup_data(data+off);
   }
}

template <typename Tm>
void qinfo3<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqinfo3: " << name << " sym=" << sym;
   tools::print_vector(dir,"dir");
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   // qblocks
   std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
   int br, bc, bm;
   for(int i=0; i<_nnzaddr.size(); i++){
      int idx = _nnzaddr[i];
      _addr_unpack(idx,br,bc,bm);
      const auto& blk = _qblocks[idx];
      if(level >= 1){
         std::cout << "idx=" << idx 
         	   << " block[" << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "," 
                   << qmid.get_sym(bm) << "]" 
                   << " dim0,dim1,dim2=(" << blk.dim0 << "," << blk.dim1 << ","
                   << blk.dim2 << ")" 
                   << std::endl; 
         if(level >= 2) blk.print("blk_"+std::to_string(idx));
      } // level>=1
   } // idx
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size() << std::endl;
   std::cout << "total size=" << _size << " sizeMB=" << tools::sizeMB<Tm>(_size) << std::endl; 
}

} // ctns

#endif
