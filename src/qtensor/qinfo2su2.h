#ifndef QINFO2SU2_H
#define QINFO2SU2_H

#include "qinfo2.h"
#include "spincoupling.h"

namespace ctns{

   template <typename Tm>
      struct qinfo2su2{
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
            // setup derived variables
            void setup();
            // conservation pattern determined by dir
            bool _ifconserve(const int br, const int bc) const{
               bool ifnele = sym.ne() == (std::get<0>(dir) ? qrow.get_sym(br).ne() : -qrow.get_sym(br).ne())
                  + (std::get<1>(dir) ? qcol.get_sym(bc).ne() : -qcol.get_sym(bc).ne());
               // triangular condition
               bool ifspin = spin_triangle(sym.ts(), qrow.get_sym(br).ts(), qcol.get_sym(bc).ts());
               return ifnele && ifspin;
            }
         public:
            // initialization
            void init(const qsym& _sym, const qbond& _qrow, const qbond& _qcol, 
                  const direction2 _dir={1,0}){
               sym = _sym;
               qrow = _qrow;
               qcol = _qcol;
               dir = _dir;
               this->setup();
            }
            // check
            bool operator ==(const qinfo2su2& info) const{
               return sym==info.sym && qrow==info.qrow && qcol==info.qcol && dir==info.dir;
            }
            // print
            void print(const std::string name) const;
            // helpers
            size_t get_offset(const int br, const int bc) const{
               return _offset.at(std::make_tuple(br,bc));
            }
            bool empty(const int br, const int bc) const{
               return this->get_offset(br,bc) == 0;
            }
            dtensor2<Tm> operator()(const int br, const int bc, Tm* data) const{
               size_t off = this->get_offset(br,bc);
               return (off == 0)? dtensor2<Tm>() : dtensor2<Tm>(qrow.get_dim(br),
                     qcol.get_dim(bc), data+off-1);
            }
         public:
            static const int dims = 2; 
            qsym sym; // <row|op[in]|col>
            qbond qrow, qcol;
            direction2 dir = dir_OPER;
            // derived
            size_t _size = 0;
            int _rows = 0, _cols = 0;
         public: 
            std::vector<std::tuple<int,int>> _nnzaddr;
            std::map<std::tuple<int,int>,size_t> _offset;
            // fast access to nonzero rows / cols
            std::vector<std::vector<int>> _br2bc, _bc2br;
      };

   template <typename Tm>
      void qinfo2su2<Tm>::setup(){
         _rows = qrow.size();
         _cols = qcol.size();
         int nblks = _rows*_cols;
         _nnzaddr.resize(nblks);
         _size = 1;
         int ndx = 0;
         for(int br=0; br<_rows; br++){
            int rdim = qrow.get_dim(br);
            for(int bc=0; bc<_cols; bc++){
               auto indices = std::make_tuple(br,bc);
               if(_ifconserve(br,bc)){
                  _nnzaddr[ndx] = indices;
                  _offset[indices] = _size;
                  int cdim = qcol.get_dim(bc);
                  _size += rdim*cdim;
                  ndx += 1;
               }else{
                  _offset[indices] = 0;
               }
            } // bc
         } // br
         _nnzaddr.resize(ndx);
         _size -= 1; // tricky part
         
         // ZL@20220621 fast access of nonzero blocks
         _br2bc.resize(_rows);
         _bc2br.resize(_cols);
         int br, bc;
         for(int i=0; i<ndx; i++){
            br = std::get<0>(_nnzaddr[i]);
            bc = std::get<1>(_nnzaddr[i]);
            _br2bc[br].push_back(bc);
            _bc2br[bc].push_back(br);
         }
      }

   template <typename Tm>
      void qinfo2su2<Tm>::print(const std::string name) const{
         std::cout << "qinfo2su2: " << name << " sym=" << sym << " dir="
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
