#ifndef MPS_CONVERT_H
#define MPS_CONVERT_H

#include "qtensor/qtensor.h"

namespace ctns{

   template <typename Qm, typename Tm>
   void site_convert(const stensor3<Tm>& site, 
         std::vector<stensor2<Tm>>& site2){
      std::cout << "NOT IMPLEMENTED YET!" << std::endl;
      exit(1);
   }

   template <>
   void site_convert<qkind::qNSz>(const stensor3<double>& site, 
         std::vector<stensor2<double>>& site2){
      const auto& qrow = site.info.qrow;
      const auto& qcol = site.info.qcol;
      const auto& qmid = site.info.qmid;
      site2[0].init(qmid.get_sym(0), qrow, qcol); // 00
      site2[1].init(qmid.get_sym(1), qrow, qcol); // 11
      site2[2].init(qmid.get_sym(2), qrow, qcol); // 01=a
      site2[3].init(qmid.get_sym(3), qrow, qcol); // 10=b
      int br, bc, bm;
      for(int i=0; i<site.info._nnzaddr.size(); i++){
         int idx = site.info._nnzaddr[i];
         site.info._addr_unpack(idx,br,bc,bm);
         const auto blk3 = site(br,bc,bm);
         auto blk2 = site2[bm](br,bc);
         size_t N = blk3.size();
         linalg::xcopy(N, blk3.data(), blk2.data());
      }
   }

} // ctns

#endif
