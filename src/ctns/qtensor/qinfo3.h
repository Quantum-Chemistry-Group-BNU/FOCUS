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
	    & _size & _exist & _qblocks;
      }
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc, const int bm) const{
	 auto qsum = -sym; // default in
	 qsum += std::get<0>(dir) ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += std::get<1>(dir) ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 qsum += std::get<2>(dir) ? qmid.get_sym(bm) : -qmid.get_sym(bm);
	 return qsum.is_zero();
      }
   public:
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
      bool ifExist(const int br, const int bc, const int bm) const{ 
         return _exist.at(std::make_tuple(br,bc,bm));
      }
      bool ifNotExist(const int br, const int bc, const int bm) const{ 
         return !_exist.at(std::make_tuple(br,bc,bm));
      }
   public:
      qsym sym;
      qbond qrow, qcol, qmid;
      direction3 dir;
      // --- derived --- 
      using index3 = std::tuple<int,int,int>; 
      std::map<index3,bool> _exist; 
      std::map<index3,std::tuple<size_t,int,int,int,int>> _qblocks;
      size_t _size;
};

template <typename Tm>
void qinfo3<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
		      const qbond& _qmid, const direction3 _dir){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   qmid = _qmid;
   dir = _dir;
   _size = 0;
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
	 for(int bm=0; bm<qmid.size(); bm++){
	    int mdim = qmid.get_dim(bm);
	    bool ifexist = _ifconserve(br,bc,bm);
	    auto key = std::make_tuple(br,bc,bm);
	    _exist[key] = ifexist;
	    if(not ifexist) continue;
	    int size = rdim*cdim*mdim;
	    _qblocks[key] = std::make_tuple(_size,size,rdim,cdim,mdim);
	    _size += size;
	 } // bm 
      } // bc
   } // br
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
   std::cout << "total no. of nonzero blocks=" << _qblocks.size() 
             << " nblocks=" << _exist.size() 
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl; 
}

} // ctns

#endif
