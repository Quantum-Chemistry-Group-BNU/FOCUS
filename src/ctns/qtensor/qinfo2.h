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
      template<class Archive>
      void serialize(Archive & ar, const unsigned int version){
	 ar & sym & qrow & qcol & dir 
	    & _size & _exist & _qblocks;
      }
      // conservation pattern determined by dir
      bool _ifconserve(const int br, const int bc) const{
	 auto qsum = -sym; // default in
	 qsum += std::get<0>(dir) ? qrow.get_sym(br) : -qrow.get_sym(br);
	 qsum += std::get<1>(dir) ? qcol.get_sym(bc) : -qcol.get_sym(bc);
	 return qsum.is_zero();
      }
   public:
      // initialization
      void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
		const direction2 _dir={1,0});
      // print
      void print(const std::string name) const;
      // check
      bool operator ==(const qinfo2& info) const{
         return sym==info.sym && qrow==info.qrow && qcol==info.qcol && dir==info.dir;
      }
      // helpers
      bool ifExist(const int br, const int bc) const{ 
         return _exist.at(std::make_tuple(br,bc));
      }
      bool ifNotExist(const int br, const int bc) const{ 
         return !_exist.at(std::make_tuple(br,bc));
      }
   public:
      qsym sym; // <row|op[in]|col>
      qbond qrow, qcol;
      direction2 dir={1,0}; // {out,int} by usual convention for operators in diagrams 
      // --- derived --- 
      using index2 = std::tuple<int,int>;
      std::map<index2,bool> _exist; 
      std::map<index2,std::tuple<size_t,int,int,int>> _qblocks;
      size_t _size;
};

template <typename Tm>
void qinfo2<Tm>::init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
	 	      const direction2 _dir){
   sym = _sym;
   qrow = _qrow;
   qcol = _qcol;
   dir = _dir;
   _size = 0;
   for(int br=0; br<qrow.size(); br++){
      int rdim = qrow.get_dim(br);
      for(int bc=0; bc<qcol.size(); bc++){
         int cdim = qcol.get_dim(bc);
         bool ifexist = _ifconserve(br,bc);
	 auto key = std::make_tuple(br,bc);
	 _exist[key] = ifexist;
	 if(not ifexist) continue;
	 int size = rdim*cdim;
	 _qblocks[key] = std::make_tuple(_size,size,rdim,cdim);
	 _size += size;
      } // bc
   } // br
}

template <typename Tm>
void qinfo2<Tm>::print(const std::string name) const{
   std::cout << "qinfo2: " << name << " sym=" << sym << " dir="
   	     << std::get<0>(dir) << "," 
	     << std::get<1>(dir) << std::endl; 
   qrow.print("qrow");
   qcol.print("qcol");
   std::cout << "total no. of nonzero blocks=" << _qblocks.size() 
             << " nblocks=" << _exist.size() 
             << " size=" << _size << ":" << tools::sizeMB<Tm>(_size) << "MB" 
             << std::endl;
}

} // ctns

#endif
