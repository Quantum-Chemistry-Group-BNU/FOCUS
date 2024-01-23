#ifndef CTNS_TOSU2_WMAT_H
#define CTNS_TOSU2_WMAT_H

#include "../init_phys.h"
#include "ctns_tosu2_qbond3.h"

namespace ctns{

   // W[alpha,beta]
   template <typename Tm>
      struct Wmatrix{
         public:
            Wmatrix(){}
            Wmatrix(const qbond& _qrow, const qbond3& _qcol);
            linalg::matrix<Tm> to_matrix() const;
         public:
            qbond qrow;
            qbond3 qcol;
            // derived
            size_t _size;
            std::map<std::pair<int,int>,size_t> _offset;
            std::vector<Tm> _data;
      };

   template <typename Tm>
      Wmatrix<Tm>::Wmatrix(const qbond& _qrow, const qbond3& _qcol){
         qrow = _qrow;
         qcol = _qcol;
         // nonzero blocks
         _size = 1;
         for(int i=0; i<qrow.size(); i++){
            auto symi = qrow.get_sym(i);
            int di = qrow.get_dim(i);
            for(int j=0; j<qcol.size(); j++){
               auto symj = qcol[j].first;
               int dj = qcol[j].second;
               // symmetry conservation
               if(symi.ne() == std::get<0>(symj) &&
                     symi.tm() == std::get<2>(symj)){
                  _offset[std::make_pair(i,j)] = _size;
                  _size += di*dj;
               }else{
                  _offset[std::make_pair(i,j)] = 0;
               }
            }
         }
         _size -= 1;
         _data.resize(_size);
         memset(_data.data(), 0, _size*sizeof(Tm));
      }
   
   template <typename Tm>   
      linalg::matrix<Tm> Wmatrix<Tm>::to_matrix() const{
         int m = qrow.get_dimAll();
         int n = get_dimAll(qcol);
         linalg::matrix<Tm> mat(m,n);
         // assign block to proper place
         auto roff = qrow.get_offset();
         auto coff = get_offset(qcol);
         for(int br=0; br<qrow.size(); br++){
            int offr = roff[br];		 
            for(int bc=0; bc<qcol.size(); bc++){
               int offc = coff[bc];
               size_t offblk = _offset.at(std::make_pair(br,bc));
               if(offblk == 0) continue;
               const Tm* ptr = _data.data() + offblk-1;
               int dimr = qrow.get_dim(br);
               int dimc = qcol[bc].second;
               for(int ic=0; ic<dimc; ic++){
                  for(int ir=0; ir<dimr; ir++){
                     mat(offr+ir,offc+ic) = *(ptr+ic*dimr+ir);
                  } // ir
               } // ic
            } // bc
         } // br
         return mat;
      }

   template <typename Tm>
      Wmatrix<Tm> initW0vac(const int ts=0, const int tm=0){
         qbond qrow = get_qbond_vac(2); 
         qbond3 qcol = get_qbond3_vac(ts); 
         Wmatrix<Tm> wmat(qrow,qcol);
         int idx = (ts+tm)/2; // Ordering: M = {-S,...,S}
         wmat._data[idx] = 1.0;
         return wmat;
      }

} // ctns

#endif
