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
                  ifspin = fock::spin_triangle(qrow.get_sym(br).ts(), qmid.get_sym(bm).ts(), tsi) && 
                     fock::spin_triangle(qcol.get_sym(bc).ts(), qver.get_sym(bv).ts(), tsj) &&
                     fock::spin_triangle(tsi, tsj, sym.ts());
               }else if(couple == LC1andC2couple){
                  ifspin = fock::spin_triangle(qrow.get_sym(br).ts(), qmid.get_sym(bm).ts(), tsi) &&
                     fock::spin_triangle(tsi, qver.get_sym(bv).ts(), tsj) && 
                     fock::spin_triangle(tsj, qcol.get_sym(bc).ts(), sym.ts()); 
               }else if(couple == C1andC2Rcouple){
                  ifspin = fock::spin_triangle(qcol.get_sym(bc).ts(), qver.get_sym(bv).ts(), tsj) &&
                     fock::spin_triangle(qmid.get_sym(bm).ts(), tsj, tsi) &&
                     fock::spin_triangle(qrow.get_sym(br).ts(), tsi, sym.ts());
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
            size_t get_offset(const int br, const int bc, const int bm, const int bv,
                 const int tsi, const int tsj) const{
               return _offset.at(std::make_tuple(br,bc,bm,bv,tsi,tsj));
            } 
            bool empty(const int br, const int bc, const int bm, const int bv, 
                  const int tsi, const int tsj) const{
               return this->get_offset(br,bc,bm,bv,tsi,tsj) == 0;
            }
            dtensor4<Tm> operator()(const int br, const int bc, const int bm, const int bv,
                  const int tsi, const int tsj, Tm* data) const{
               size_t off = this->get_offset(br,bc,bm,bv,tsi,tsj);
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
         _rows = qrow.size();
         _cols = qcol.size();
         _mids = qmid.size();
         _vers = qver.size();
         _size = 1;
         for(int br=0; br<_rows; br++){
            int rdim = qrow.get_dim(br);
            for(int bc=0; bc<_cols; bc++){
               int cdim = qcol.get_dim(bc);
               int rcdim = rdim*cdim;
               for(int bm=0; bm<_mids; bm++){
                  int mdim = qmid.get_dim(bm);
                  int rcmdim = rcdim*mdim;
                  for(int bv=0; bv<_vers; bv++){
                     int vdim = qver.get_dim(bv);
                     // different coupling cases
                     int tsl = qrow.get_sym(br).ts();
                     int tsc1 = qmid.get_sym(bm).ts();
                     int tsc2 = qver.get_sym(bv).ts();
                     int tsr = qcol.get_sym(bc).ts();
                     if(couple == LC1andC2Rcouple){
                        // sLsC1 => sLC1, sC2sR => sC2R
                        for(int tslc1=std::abs(tsl-tsc1); tslc1<=tsl+tsc1; tslc1+=2){
                           for(int tsc2r=std::abs(tsc2-tsr); tsc2r<=tsc2+tsr; tsc2r+=2){
                              auto indices = std::make_tuple(br,bc,bm,bv,tslc1,tsc2r);
                              if(_ifconserve(br,bc,bm,tslc1,tsc2r)){
                                 _nnzaddr.push_back(indices);
                                 _offset[indices] = _size;
                                 _size += rcmdim*vdim;
                              }else{
                                 _offset[indices] = 0;
                              }
                           }
                        }
                     }else if(couple == LC1andC2couple){
                        // sLsC1 => sLC1, sLC1sC2 => sLC1C2
                        for(int tslc1=std::abs(tsl-tsc1); tslc1<=tsl+tsc1; tslc1+=2){
                           for(int tslc1c2=std::abs(tslc1-tsc2); tslc1c2<=tslc1+tsc2; tslc1c2+=2){
                              auto indices = std::make_tuple(br,bc,bm,bv,tslc1,tslc1c2);
                              if(_ifconserve(br,bc,bm,tslc1,tslc1c2)){
                                 _nnzaddr.push_back(indices);
                                 _offset[indices] = _size;
                                 _size += rcmdim*vdim;
                              }else{
                                 _offset[indices] = 0;
                              }
                           }
                        }
                     }else if(couple == C1andC2Rcouple){
                        // sC2sR => sC2R, sC1sC2R => sC1C2R
                        for(int tsc2r=std::abs(tsc2-tsr); tsc2r<=tsc2+tsr; tsc2r+=2){
                           for(int tsc1c2r=std::abs(tsc1-tsc2r); tsc1c2r<=tsc1+tsc2r; tsc1c2r+=2){
                              auto indices = std::make_tuple(br,bc,bm,bv,tsc2r,tsc1c2r);
                              if(_ifconserve(br,bc,bm,tsc2r,tsc1c2r)){
                                 _nnzaddr.push_back(indices);
                                 _offset[indices] = _size;
                                 _size += rcmdim*vdim;
                              }else{
                                 _offset[indices] = 0;
                              }
                           }
                        }
                     } // couple
                  } // bv
               } // bm
            } // bc
         } // br
         _size -= 1; // tricky part
      }

   template <typename Tm>
      void qinfo4su2<Tm>::print(const std::string name) const{
         std::cout << "qinfo4su2: " << name << " sym=" << sym 
            << " couple=" << couple
            << std::endl;
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
