#ifndef CTNS_TOSU2_MSITE_H
#define CTNS_TOSU2_MSITE_H

#include "ctns_tosu2_qbond3.h"

namespace ctns{

   template <typename Tm>
      struct MixedRSite{
         public:
            MixedRSite(const qbond& _qrow, const qbond3& _qcol, const qbond3& _qmid);
         public:
            qbond qrow;
            qbond3 qcol, qmid;
            // derived
            size_t _size;
            std::map<std::tuple<int,int,int>,size_t> _offset; // (i,j,k)->offset
            std::vector<Tm> _data;
      };

   template <typename Tm>
      MixedRSite<Tm>::MixedRSite(const qbond& _qrow, const qbond3& _qcol, const qbond3& _qmid){
         qrow = _qrow;
         qcol = _qcol;
         qmid = _qmid;
         // nonzero blocks;
         _size = 1;
         for(int i=0; i<qrow.size(); i++){
            auto symi = qrow.get_sym(i);
            int di = qrow.get_dim(i);
            for(int j=0; j<qcol.size(); j++){
               auto symj = qcol[j].first;
               int dj = qcol[j].second;
               for(int k=0; k<qmid.size(); k++){
                  auto symk = qmid[k].first;
                  int dk = qmid[k].second;
                  // symmetry conservation
                  if(symi.ne() == std::get<0>(symj)+std::get<0>(symk) &&
                        symi.tm() == std::get<2>(symj)+std::get<2>(symk)){
                     _offset[std::make_tuple(i,j,k)] = _size;
                     _size += di*dj*dk;
                  }else{
                     _offset[std::make_tuple(i,j,k)] = 0;
                  }
               }
            }
         }
         _size -= 1;
         _data.resize(_size);
         memset(_data.data(), 0, _size*sizeof(Tm));
      }

   // A[l,r,m]*w[r,r'] => Am[l,r',m], similar to contract_qt3_qt2_r
   template <typename Tm>
      MixedRSite<Tm> formMixedRSite(const stensor3<Tm>& rsite,
            const Wmatrix<Tm>& wmat,
            const bool debug=true){
         const Tm alpha = 1.0, beta = 1.0;
         if(debug) std::cout << "\nctns::formMixedRSite" << std::endl;
         // init Am 
         const auto& qrow = rsite.info.qrow;
         const auto& qcol = wmat.qcol;
         const auto qmid = get_qbond_phys(); 
         MixedRSite<Tm> msite(qrow, qcol, qmid);
         // start contraction 
         int br, bc, bm;
         for(const auto& pr : msite._offset){
            size_t off3 = pr.second;
            if(off3 == 0) continue;
            Tm* blk3 = msite._data.data() + off3-1;
            br = std::get<0>(pr.first);
            bc = std::get<1>(pr.first);
            bm = std::get<2>(pr.first);
            int rdim = msite.qrow.get_dim(br);
            int cdim = msite.qcol[bc].second;
            int mdim = msite.qmid[bm].second;
            size_t size = rdim*cdim*mdim;
            // find contracted index for
            // qt3(r,c,m) = \sum_x qt3a(r,x,m)*qt2(x,c)
            for(int bx=0; bx<wmat.qrow.size(); bx++){
               size_t off3a = rsite.info._offset[rsite.info._addr(br,bx,bm)];
               size_t off2 = wmat._offset.at(std::make_pair(bx,bc));
               if(off3a == 0 || off2 == 0) continue;
               const Tm* blk3a = rsite.data() + off3a-1;
               const Tm* blk2 = wmat._data.data() + off2-1;
               int xdim = rsite.info.qcol.get_dim(bx);
               assert(xdim = wmat.qrow.get_dim(bx));
               int LDB = xdim;
               int rcdim = rdim*cdim;
               int rxdim = rdim*xdim;
               for(int im=0; im<mdim; im++){
                  const Tm* blk3a_im = blk3a + im*rxdim;
                  Tm* blk3_im = blk3 + im*rcdim;
                  linalg::xgemm("N", "N", rdim, cdim, xdim, alpha,
                        blk3a_im, rdim, blk2, LDB, beta,
                        blk3_im, rdim);
               } // im
            }
         } // i
         return msite;
      }

   // formula: qt2(r,c) = \sum_xm Conj[qt3a](r,x,m)*qt3b(c,x,m)
   template <typename Tm>
      stensor2<Tm> contract_qt3_qt3_cr(const MixedRSite<Tm>& qt3a,
            const MixedRSite<Tm>& qt3b){
         const Tm alpha = 1.0, beta = 1.0;
         stensor2<Tm> qt2(qsym({2,0,0}), qt3a.qrow, qt3b.qrow, {0,1});
         // loop over qt3a
         int br, bx, bm;
         for(const auto& pr : qt3a._offset){
            const auto& triplet = pr.first;
            const auto& off3a = pr.second;
            if(off3a == 0) continue;
            br = std::get<0>(triplet);
            bx = std::get<1>(triplet);
            bm = std::get<2>(triplet);
            const Tm* blk3a = qt3a._data.data() + off3a-1;
            int rdim = qt3a.qrow.get_dim(br);
            int xdim = qt3a.qcol[bx].second;
            int mdim = qt3a.qmid[bm].second;
            // loop over bc
            for(int bc=0; bc<qt2.cols(); bc++){
               size_t off3b = qt3b._offset.at(std::make_tuple(bc,bx,bm));
               if(off3b == 0) continue;
               size_t off2 = qt2.info._offset[qt2.info._addr(br,bc)];
               if(off2 == 0) continue;
               // qt2(r,c) = Conj[qt3a](r,x,m)*qt3b(c,x,m)
               const Tm* blk3b = qt3b._data.data() + off3b-1;
               Tm* blk2 = qt2._data + off2-1;
               int cdim = qt2.info.qcol.get_dim(bc);
               int xmdim = xdim*mdim;
               linalg::xgemm("N", "C", rdim, cdim, xmdim, alpha,
                     blk3a, rdim, blk3b, cdim, beta,
                     blk2, rdim);
            } // bc
         } // i
         linalg::xconj(qt2.info._size, qt2._data);
         return qt2;
      }

} // ctns

#endif
