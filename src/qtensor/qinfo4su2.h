#ifndef QINFO4SU2_H
#define QINFO4SU2_H

#include "qinfo4.h"
#include "spincoupling.h"

namespace ctns{

   template <typename Tm>
      struct qinfo4su2{
         private:
            // serialize
            friend class boost::serialization::access;	   
            template <class Archive>
               void save(Archive & ar, const unsigned int version) const{
                  ar & sym & qrow & qcol & qmid & qver & couple;
               }
            template <class Archive>
               void load(Archive & ar, const unsigned int version){
                  ar & sym & qrow & qcol & qmid & qver & couple;
                  this->setup();
               }
            BOOST_SERIALIZATION_SPLIT_MEMBER()
               // setup derived variables
               void setup();
            // conservation: dir={1,1,1,1} 
            bool _ifconserve(const int br, const int bc, const int bm, const int bv, 
                  const int tsi, const int tsj) const{
               bool ifnele = sym.ne() == qrow.get_sym(br).ne() + qcol.get_sym(bc).ne() 
                  + qmid.get_sym(bm).ne() + qver.get_sym(bv).ne();
               // triangular condition
               bool ifspin;
               if(couple == LC1andC2Rcouple){
                  ifspin = spin_triangle(qrow.get_sym(br).tm(), qmid.get_sym(bm).tm(), tsi) && 
                     spin_triangle(qcol.get_sym(bc).tm(), qver.get_sym(bv).tm(), tsj) &&
                     spin_triangle(tsi, tsj, sym.tm());
               }else if(couple == LC1andC2couple){
                  ifspin = spin_triangle(qrow.get_sym(br).tm(), qmid.get_sym(bm).tm(), tsi) &&
                     spin_triangle(tsi, qver.get_sym(bv).tm(), tsj) && 
                     spin_triangle(tsj, qcol.get_sym(bc).tm(), sym.tm()); 
               }else if(couple == C1andC2Rcouple){
                  ifspin = spin_triangle(qcol.get_sym(bc).tm(), qver.get_sym(bv).tm(), tsj) &&
                     spin_triangle(qmid.get_sym(bm).tm(), tsj, tsi) &&
                     spin_triangle(qrow.get_sym(br).tm(), tsi, sym.tm());
               }
               return ifnele && ifspin; 
            }
         public:
            // initialization
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol,
                  const qbond& _qmid, const qbond& _qver, 
                  const spincoupling4 _couple=LC1andC2Rcouple){
               sym = _sym;
               qrow = _qrow;
               qcol = _qcol;
               qmid = _qmid;
               qver = _qver;
               couple = _couple;
               this->setup(); 
            }
            // print
            void print(const std::string name) const;
            // check
            bool operator ==(const qinfo4su2& info) const{
               return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
                  && qmid==info.qmid && qver==info.qver && couple=info.couple;
            }
            // helpers
            bool empty(const int br, const int bc, const int bm, const int bv, 
                  const int tsi, const int tsj) const{
               return _offset.at(std::make_tuple(br,bc,bm,bv,tsi,tsj)) == 0;
            }
            dtensor4<Tm> operator()(const int br, const int bc, const int bm, const int bv,
                  const int tsi, const int tsj, Tm* data) const{
               size_t off = _offset.at(std::make_tuple(br,bc,bm,bv,tsi,tsj));
               return (off == 0)? dtensor4<Tm>() : dtensor4<Tm>(qrow.get_dim(br),
                     qcol.get_dim(bc), qmid.get_dim(bm), qver.get_dim(bv), data+off-1);
            }
         public:
            static const int dims = 4; 
            qsym sym;
            qbond qrow, qcol, qmid, qver;
            spincoupling4 couple;
            // derived
            size_t _size = 0;
            int _rows = 0, _cols = 0, _mids = 0, _vers = 0;
         public: 
            std::vector<std::tuple<int,int,int,int,int,int>> _nnzaddr; // (br,bc,bm,bv,tsi,tsj)
            std::map<std::tuple<int,int,int,int,int,int>,size_t> _offset;
      };

   template <typename Tm>
      void qinfo4su2<Tm>::setup(){
         /*
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
            */
      }

   template <typename Tm>
      void qinfo4su2<Tm>::print(const std::string name) const{
         std::cout << "qinfo4su2: " << name << " sym=" << sym << std::endl;
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
