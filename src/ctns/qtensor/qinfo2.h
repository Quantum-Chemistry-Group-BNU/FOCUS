#ifndef QINFO2_H
#define QINFO2_H

#include "../../core/serialization.h"
#include "../../core/tools.h"

namespace ctns{

using direction2 = std::tuple<bool,bool>;

template <typename Tm>
struct qinfo2{
   private:
      // serialize
      friend class boost::serialization::access;	   
      template <class Archive>
      void save(Archive & ar, const unsigned int version) const{
	 ar & sym & qrow & qcol & dir;
      }
      template <class Archive>
      void load(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & dir;
	 this->setup();
      }
      BOOST_SERIALIZATION_SPLIT_MEMBER()
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc) const{
	 return sym == (std::get<0>(dir) ? qrow.get_sym(br) : -qrow.get_sym(br))
		     + (std::get<1>(dir) ? qcol.get_sym(bc) : -qcol.get_sym(bc));
      }
      // setup derived variables
      void setup();
   public:
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
		const direction2 _dir={1,0}){
         sym = _sym;
         qrow = _qrow;
         qcol = _qcol;
         dir = _dir;
         this->setup();
      }
      // print
      void print(const std::string name) const;
      // check
      bool operator ==(const qinfo2& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol && dir==info.dir;
      }
      // helpers
      bool empty(const int br, const int bc) const{
	 return _offset[_addr(br,bc)] == 0;
      }
      dtensor2<Tm> operator()(const int br, const int bc, 
		      	      Tm* data) const{
         size_t off = _offset[_addr(br,bc)];
	 return (off == 0)? dtensor2<Tm>() : dtensor2<Tm>(qrow.get_dim(br),
		  	        			  qcol.get_dim(bc),
				  		          data+off-1);
      }
   public:
      qsym sym; // <row|op[in]|col>
      qbond qrow, qcol;
      direction2 dir={1,0}; // {out,int} by usual convention for operators in diagrams 
   public: // derived
      size_t _size;
      int _rows, _cols;
      std::vector<int> _nnzaddr;
      std::vector<size_t> _offset;
};

template <typename Tm>
void qinfo2<Tm>::setup(){
   _rows = qrow.size();
   _cols = qcol.size();
   int nblks = _rows*_cols;
   _nnzaddr.resize(nblks);
   _offset.resize(nblks, 0);
   _size = 1;
   int idx = 0, ndx = 0;
   for(int br=0; br<_rows; br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<_cols; bc++){
         if(_ifconserve(br,bc)){
	    _nnzaddr[ndx] = idx;
	    _offset[idx] = _size;
	    int cdim = qcol.get_dim(bc);
	    _size += rdim*cdim;
	    ndx += 1;
	 }
	 idx += 1;
      } // bc
   } // br
   _nnzaddr.resize(ndx);
   _size -= 1; // tricky part
}

template <typename Tm>
void qinfo2<Tm>::print(const std::string name) const{
   std::cout << "qinfo2: " << name << " sym=" << sym << " dir="
   	     << std::get<0>(dir) << "," 
	     << std::get<1>(dir) << std::endl; 
   qrow.print("qrow");
   qcol.print("qcol");
   std::cout << "total no. of nonzero blocks=" << _nnzaddr.size()
             << " nblocks=" << _offset.size()
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl;
}

} // ctns

#endif
