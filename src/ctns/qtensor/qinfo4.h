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
	    & _size & _exist & _qblocks; 
      }
      // conservation: dir={1,1,1,1} 
      bool _ifconserve(const int br, const int bc, const int bm, const int bv) const{
	 return sym == qrow.get_sym(br) + qcol.get_sym(bc) 
		     + qmid.get_sym(bm) + qver.get_sym(bv);
      }
   public:
      // initialization
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
	        const qbond& _qmid, const qbond& _qver);
      // print
      void print(const std::string name) const;
      // check
      bool operator ==(const qinfo4& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
		 && qmid==info.qmid && qver==info.qver;
      }
      // helpers
      bool ifExist(const int br, const int bc, const int bm, const int bv) const{ 
         return _exist.at(std::make_tuple(br,bc,bm,bv));
      }
      bool ifNotExist(const int br, const int bc, const int bm, const int bv) const{ 
         return !_exist.at(std::make_tuple(br,bc,bm,bv));
      }
   public:
      qsym sym;
      qbond qrow, qcol, qmid, qver;
      // --- derived --- 
      using index4 = std::tuple<int,int,int,int>; 
      std::map<index4,bool> _exist; 
      std::map<index4,std::tuple<size_t,int,int,int,int,int>> _qblocks;
      size_t _size;
};

template <typename Tm>
void qinfo4<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
		      const qbond& _qmid, const qbond& _qver){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   qmid = _qmid;
   qver = _qver;
   _size = 0;
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
	 for(int bm=0; bm<qmid.size(); bm++){
	    int mdim = qmid.get_dim(bm);
	    for(int bv=0; bv<qver.size(); bv++){
	       int vdim = qver.get_dim(bv);
	       bool ifexist = _ifconserve(br,bc,bm,bv);
	       auto key = std::make_tuple(br,bc,bm,bv);
	       _exist[key] = ifexist;
	       if(not ifexist) continue;
	       int size = rdim*cdim*mdim*vdim;
	       _qblocks[key] = std::make_tuple(_size,size,rdim,cdim,mdim,vdim);
	       _size += size;
	    } // bv
	 } // bm
      } // bc
   } // br
}

template <typename Tm>
void qinfo4<Tm>::print(const std::string name) const{
   std::cout << "qinfo4: " << name << " sym=" << sym << std::endl;
   qrow.print("qrow");
   qcol.print("qcol");
   qmid.print("qmid");
   qver.print("qver");
   std::cout << "total no. of nonzero blocks=" << _qblocks.size() 
             << " nblocks=" << _exist.size() 
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl; 
}

} // ctns

#endif
