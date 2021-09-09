#ifndef TENSOR_QINFO2_H
#define TENSOR_QINFO2_H

#include "../../core/serialization.h"
#include "../qtensor/qtensor.h"

namespace ctns{

template <typename Tm>
struct qinfo2{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & dir 
	    & _rows & _cols & _size & _qblocks;
      }
   public:
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += dir[0] ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += dir[1] ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum.is_zero();
      }
      // address for storaging block data 
      int _addr(const int br, const int bc) const{ return br*_cols + bc; }
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, const std::vector<bool> _dir={1,0});
      void setup_qblocks(const Tm* data);
   public:
      qsym sym;
      qbond qrow, qcol;
      std::vector<bool> dir={1,0};	    
   public:
      int _rows, _cols;
      size_t _size = 0;
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
   for(int br=0; br<qrow.size(); br++){
      for(int bc=0; bc<qcol.size(); bc++){
	 if(not _ifconserve(br,bc)) continue;
         int rdim = qrow.get_dim(br);
         int cdim = qcol.get_dim(bc);
	 _qblocks[_addr(br,br)].setup_dims(rdim,cdim,_size);
	 _size += rdim*cdim;
      } // bc
   } // br
}

template <typename Tm>
void qinfo2<Tm>::setup_qblocks(const Tm* data){
   for(int br=0; br<qrow.size(); br++){
      for(int bc=0; bc<qcol.size(); bc++){
         auto& blk = _qblocks[_addr(br,bc)];
	 if(blk.size() == 0) continue;
	 blk.setup_data(data);
      } // bc
   } // br
}

} // ctns

#endif
