#ifndef QINFO3SU2_H
#define QINFO3SU2_H

#include "qinfo3.h"
#include "spincoupling.h"

namespace ctns{

   template <typename Tm>
      struct qinfo3su2{
         private:
            // serialize
            friend class boost::serialization::access;	   
            template <class Archive>
               void save(Archive & ar, const unsigned int version) const{
                  ar & sym & qrow & qcol & qmid & dir & couple;
               }
            template <class Archive>
               void load(Archive & ar, const unsigned int version){
                  ar & sym & qrow & qcol & qmid & dir & couple;
                  this->setup();
               }
            BOOST_SERIALIZATION_SPLIT_MEMBER()
               // setup derived variables
               void setup();
            // conservation pattern determined by dir
            bool _ifconserve(const int br, const int bc, const int bm, const int tsi) const{
               bool ifnele = sym.ne() == (std::get<0>(dir) ? qrow.get_sym(br).ne() : -qrow.get_sym(br).ne())
                  + (std::get<1>(dir) ? qcol.get_sym(bc).ne() : -qcol.get_sym(bc).ne())
                  + (std::get<2>(dir) ? qmid.get_sym(bm).ne() : -qmid.get_sym(bm).ne());
               // triangular condition
               bool ifspin;
               if(couple == LCcouple){
                  ifspin = fock::spin_triangle(qrow.get_sym(br).ts(), qmid.get_sym(bm).ts(), tsi) &&
                     fock::spin_triangle(tsi, qcol.get_sym(bc).ts(), sym.ts());
               }else if(couple == CRcouple){
                  ifspin = fock::spin_triangle(qmid.get_sym(bm).ts(), qcol.get_sym(bc).ts(), tsi) &&
                     fock::spin_triangle(tsi, qrow.get_sym(br).ts(), sym.ts());
               }
               return ifnele && ifspin;
            }
         public:
            // initialization
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
                  const qbond& _qmid, const direction3 _dir=dir_RCF, 
                  const spincoupling3 _couple=CRcouple){
               sym = _sym;
               qrow = _qrow;
               qcol = _qcol;
               qmid = _qmid;
               dir = _dir;
               couple = _couple;
               this->setup();
            }
            // print
            void print(const std::string name) const;
            // check
            bool operator ==(const qinfo3su2& info) const{
               return sym==info.sym && qrow==info.qrow && qcol==info.qcol 
                  && qmid==info.qmid && dir==info.dir 
                  && couple=info.couple;
            }
            // helpers
            size_t get_offset(const int br, const int bc, const int bm, const int tsi) const{
               auto key = std::make_tuple(br,bc,bm,tsi);
               if(_offset.find(key) == _offset.end()){
                  return 0;
               }else{
                  return _offset.at(std::make_tuple(br,bc,bm,tsi));
               }
            }
            bool empty(const int br, const int bc, const int bm, const int tsi) const{
               return this->get_offset(br,bc,bm,tsi) == 0;
            }
            dtensor3<Tm> operator()(const int br, const int bc, const int bm, 
                  const int tsi, Tm* data) const{
               size_t off = this->get_offset(br,bc,bm,tsi);
               return (off == 0)? dtensor3<Tm>() : dtensor3<Tm>(qrow.get_dim(br),
                     qcol.get_dim(bc), qmid.get_dim(bm), data+off-1);
            }
            // ZL@20221207 for dump MPS RCF site into binary format [sym,couple can be deduced] 
            void dump(std::ofstream& ofs) const;
         public:
            static const int dims = 3;
            qsym sym;
            qbond qrow, qcol, qmid;
            direction3 dir;
            spincoupling3 couple;
            // derived
            size_t _size = 0;
            int _rows = 0, _cols = 0, _mids = 0;
         public: 
            std::vector<std::tuple<int,int,int,int>> _nnzaddr; // (br,bc,bm,tsi)
            std::map<std::tuple<int,int,int,int>,size_t> _offset;
      };

   template <typename Tm>
      void qinfo3su2<Tm>::dump(std::ofstream& ofs) const{
         // C order
         qrow.dump(ofs);
         qcol.dump(ofs);
         qmid.dump(ofs);
         ofs.write((char*)(_nnzaddr.data()), sizeof(_nnzaddr[0])*_nnzaddr.size());
         ofs.write((char*)(&_size), sizeof(_size)); // F order
      }

   template <typename Tm>
      void qinfo3su2<Tm>::setup(){
         _rows = qrow.size();
         _cols = qcol.size();
         _mids = qmid.size();
         _size = 1;
         for(int br=0; br<_rows; br++){
            int rdim = qrow.get_dim(br);
            for(int bc=0; bc<_cols; bc++){
               int cdim = qcol.get_dim(bc);
               int rcdim = rdim*cdim;
               for(int bm=0; bm<_mids; bm++){
                  int mdim = qmid.get_dim(bm);
                  // different coupling cases
                  int tsl = qrow.get_sym(br).ts();
                  int tsc = qmid.get_sym(bm).ts();
                  int tsr = qcol.get_sym(bc).ts();
                  if(couple == LCcouple){
                     // sLsC => sLC
                     for(int tslc=std::abs(tsl-tsc); tslc<=tsl+tsc; tslc+=2){
                        auto indices = std::make_tuple(br,bc,bm,tslc); 
                        if(_ifconserve(br,bc,bm,tslc)){
                           _nnzaddr.push_back(indices);
                           _offset[indices] = _size;
                           _size += rcdim*mdim;
                        }else{
                           _offset[indices] = 0;
                        }
                     }
                  }else if(couple == CRcouple){
                     // sCsR => sCR
                     for(int tscr=std::abs(tsc-tsr); tscr<=tsc+tsr; tscr+=2){
                        auto indices = std::make_tuple(br,bc,bm,tscr);
                        if(_ifconserve(br,bc,bm,tscr)){
                           _nnzaddr.push_back(indices);
                           _offset[indices] = _size;
                           _size += rcdim*mdim;
                        }else{
                           _offset[indices] = 0;
                        }
                     }
                  } // couple
               } // bm 
            } // bc
         } // br
         _size -= 1; // tricky part
      }

   template <typename Tm>
      void qinfo3su2<Tm>::print(const std::string name) const{
         std::cout << "qinfo3su2: " << name << " sym=" << sym << " dir="
            << std::get<0>(dir) << "," 
            << std::get<1>(dir) << ","
            << std::get<2>(dir) 
            << " couple=" << couple
            << std::endl; 
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
