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
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & sym & qrow & qcol & qmid & qver;
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & qmid & qver;
	 this->setup_qblocks();
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
      // conservation: dir={1,1,1,1} 
      bool _ifconserve(const int br, const int bc, const int bm, const int bv) const{
	 return sym == qrow.get_sym(br) 
		     + qcol.get_sym(bc) 
		     + qmid.get_sym(bm) 
		     + qver.get_sym(bv);
      }
      // setup derived variables
      void setup();
   public:
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
	        const qbond& _qmid, const qbond& _qver){
         sym = _sym;
         qrow = _qrow;
         qcol = _qcol;
	 qmid = _qmid;
         qver = _qver;
	 this->setup(); 
      }
      // print
      void print(const std::string name) const;
      // check
      bool operator ==(const qinfo4& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
		 && qmid==info.qmid && qver==info.qver;
      }
      // helpers
      bool empty(const int br, const int bc, const int bm, const int bv) const{
	 return _offset[_addr(br,bc,bm,bv)] == 0;
      }
      dtensor4<Tm> operator()(const int br, const int bc, const int bm, const int bv,
		      	      Tm* data) const{
         size_t off = _offset[_addr(br,bc,bm,bv)];
	 return (off == 0)? dtensor4<Tm>() : dtensor4<Tm>(qrow.get_dim(br),
		  	        			  qcol.get_dim(bc),
		  	        			  qmid.get_dim(bm),
		  	        			  qver.get_dim(bv),
				  		          data+off-1);
      }
   public:
      static const int dims = 4; 
      qsym sym;
      qbond qrow, qcol, qmid, qver;
   public: // derived
      size_t _size = 0;
      int _rows = 0, _cols = 0, _mids = 0, _vers = 0;
      std::vector<int> _nnzaddr;
      std::vector<size_t> _offset;
};

template <typename Tm>
void qinfo4<Tm>::setup(){
   _rows = qrow.size();
   _cols = qcol.size();
   _mids = qmid.size();
   _vers = qver.size();
   int nblks = _rows*_cols*_mids*_vers;
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
	    int mdim = qmid.get_dim(bm);
	    int rcmdim = rcdim*mdim;
	    for(int bv=0; bv<_vers; bv++){
	       if(_ifconserve(br,bc,bm,bv)){
		  _nnzaddr[ndx] = idx;
		  _offset[idx] = _size;
	          int vdim = qver.get_dim(bv);
		  _size += rcmdim*vdim;
		  ndx += 1;
	       }
	       idx += 1;
	    } // bv
	 } // bm
      } // bc
   } // br
   _nnzaddr.resize(ndx);
   _size -= 1; // tricky part
}

template <typename Tm>
void qinfo4<Tm>::print(const std::string name) const{
   std::cout << "qinfo4: " << name << " sym=" << sym << std::endl;
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   qver.print("qver");
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size()
             << " nblocks=" << _offset.size()
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl; 
}

} // ctns

#endif
