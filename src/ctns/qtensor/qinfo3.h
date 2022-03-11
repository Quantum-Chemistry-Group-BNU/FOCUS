#ifndef QINFO3_H
#define QINFO3_H

#include "../../core/serialization.h"
#include "../../core/tools.h"

namespace ctns{

using direction3 = std::tuple<bool,bool,bool>;
const direction3 dir_RCF = {0,1,1}; // RCF (default)
const direction3 dir_LCF = {1,0,1}; // LCF
const direction3 dir_CCF = {1,1,0}; // CCF (for internal upward node)
const direction3 dir_WF3 = {1,1,1}; // WF
extern const direction3 dir_RCF;
extern const direction3 dir_LCF;
extern const direction3 dir_CCF;
extern const direction3 dir_WF3;

template <typename Tm>
struct qinfo3{
   private:
      // serialize
      friend class boost::serialization::access;
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & qmid & dir
	    & _size & _rows & _cols & _mids
	    & _nnzaddr & _offset;
      }
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc, const int bm) const{
	 return sym == (std::get<0>(dir) ? qrow.get_sym(br) : -qrow.get_sym(br))
	 	     + (std::get<1>(dir) ? qcol.get_sym(bc) : -qcol.get_sym(bc))
	 	     + (std::get<2>(dir) ? qmid.get_sym(bm) : -qmid.get_sym(bm));
      }
   public:
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
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
		const qbond& _qmid, const direction3 _dir=dir_RCF);
      // print
      void print(const std::string name) const;
      // check
      bool operator ==(const qinfo3& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
		 && qmid==info.qmid && dir==info.dir;
      }
      // helpers
      bool empty(const int br, const int bc, const int bm) const{
	 return _offset[_addr(br,bc,bm)] == 0;
      }
      dtensor3<Tm> get_dtensor(const int br, const int bc, const int bm, 
		      	       Tm* data) const{
         size_t off = _offset[_addr(br,bc,bm)];
	 return (off == 0)? dtensor3<Tm>() : dtensor3<Tm>(qrow.get_dim(br),
		  	        			  qcol.get_dim(bc),
		  	        			  qmid.get_dim(bm),
				  		          data+off-1);
      }
   public:
      qsym sym;
      qbond qrow, qcol, qmid;
      direction3 dir;
      // --- derived --- 
      size_t _size;
      int _rows, _cols, _mids;
      std::vector<int> _nnzaddr;
      std::vector<size_t> _offset;
};

template <typename Tm>
void qinfo3<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
		      const qbond& _qmid, const direction3 _dir){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   qmid = _qmid;
   dir = _dir;
   _rows = qrow.size();
   _cols = qcol.size();
   _mids = qmid.size();
   int nblks = _rows*_cols*_mids;
   _nnzaddr.resize(nblks);
   _offset.resize(nblks, 0);
   _size = 1;
   int idx = 0, ndx = 0;
   for(int br=0; br<_rows; br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<_cols; bc++){
         int cdim = qcol.get_dim(bc);
	 int rcdim = rdim*cdim;
	 for(int bm=0; bm<_mids; bm++){
	    if(_ifconserve(br,bc,bm)){
	       _nnzaddr[ndx] = idx;
	       _offset[idx] = _size;
	       int mdim = qmid.get_dim(bm);
	       _size += rcdim*mdim;
	       ndx += 1;
	    }
	    idx += 1;
	 } // bm 
      } // bc
   } // br
   _nnzaddr.resize(ndx);
   _size -= 1; // tricky part
}

template <typename Tm>
void qinfo3<Tm>::print(const std::string name) const{
   std::cout << "qinfo3: " << name << " sym=" << sym << " dir="
   	     << std::get<0>(dir) << "," 
	     << std::get<1>(dir) << ","
	     << std::get<2>(dir) << std::endl; 
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size()
             << " nblocks=" << _offset.size()
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl; 
}

} // ctns

#endif
