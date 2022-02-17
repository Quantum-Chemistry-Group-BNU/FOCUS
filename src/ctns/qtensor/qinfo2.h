#ifndef QINFO2_H
#define QINFO2_H

#include "../../core/serialization.h"
#include "../../core/tools.h"

namespace ctns{

// SPARSE TENSOR: symmetry block is in C ORDER

template <typename Tm>
struct qinfo2{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & dir 
	    & _rows & _cols & _size 
	    & _nnzaddr & _qblocks;
      }
   public:
      // --- GENERAL FUNCTIONS ---
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[1] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum.is_zero();
      }
      // address for storaging block data  - FORTRAN ORDER
      int _addr(const int br, const int bc) const{ 
	 return br*_cols + bc; 
      }
      void _addr_unpack(const int idx2, int& br, int& bc) const{
	 bc = idx2%_cols;
         br = idx2/_cols;
      }
      // initialization
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
		const std::vector<bool> _dir={1,0});
      void setup_data(Tm* data);
      // print
      void print(const std::string name, const int level=0) const;
      // --- SPECIFIC FUNCTIONS ---
      bool operator ==(const qinfo2& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol && dir==info.dir;
      }
   public:
      qsym sym; // <row|op[in]|col>
      qbond qrow, qcol;
      std::vector<bool> dir={1,0}; // {out,int} by usual convention for operators in diagrams 
      // --- derived --- 
      int _rows, _cols;
      size_t _size = 0;
      std::vector<int> _nnzaddr;
      std::vector<dtensor2<Tm>> _qblocks;
};

template <typename Tm>
void qinfo2<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
	 	      const std::vector<bool> _dir){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   dir = _dir;
   _rows = qrow.size();
   _cols = qcol.size();
   _qblocks.resize(_rows*_cols);
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
	 if(not _ifconserve(br,bc)) continue;
         int cdim = qcol.get_dim(bc);
         int addr = _addr(br,bc);
         _nnzaddr.push_back(addr);
	 _qblocks[addr].setup_dims(rdim,cdim,_size);
	 _size += rdim*cdim;
      } // bc
   } // br
}

template <typename Tm>
void qinfo2<Tm>::setup_data(Tm* data){
   for(int i=0; i<_nnzaddr.size(); i++){
      int addr = _nnzaddr[i];
      _qblocks[addr].setup_data(data);
   }
}

template <typename Tm>
void qinfo2<Tm>::print(const std::string name, const int level) const{
   std::cout << "\nqinfo2: " << name << " sym=" << sym;
   tools::print_vector(dir,"dir");
   qrow.print("qrow");
   qcol.print("qcol");
   // qblocks
   std::cout << "qblocks: nblocks=" << _qblocks.size() << std::endl;
   int br, bc;
   for(int i=0; i<_nnzaddr.size(); i++){
      int idx = _nnzaddr[i];
      _addr_unpack(idx,br,bc);
      const auto& blk = _qblocks[idx];
      if(level >= 1){
         std::cout << "idx=" << idx 
         	   << " block[" << qrow.get_sym(br) << "," << qcol.get_sym(bc) << "]" 
                   << " dim0,dim1=(" << blk.dim0 << "," << blk.dim1 << ")" 
                   << std::endl; 
         if(level >= 2) blk.print("blk_"+std::to_string(idx));
      } // level>=1
   } // idx
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size() << std::endl;
   std::cout << "total size=" << _size << " sizeMB=" << tools::sizeMB<Tm>(_size) << std::endl; 
}

} // ctns

#endif
