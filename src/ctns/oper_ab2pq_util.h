#ifndef OPER_AB2PQ_UTIL_H
#define OPER_AB2PQ_UTIL_H

#include "sadmrg/symbolic_compxwf_opS_su2.h"

namespace ctns{

   template <typename Tm>
      linalg::matrix<Tm> get_A2Pmat(const std::vector<int>& aindex, 
            const std::vector<int>& pindex, 
            const integral::two_body<Tm>& int2e){
         int rows = aindex.size();
         int cols = pindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ipq = pindex[icol];
            auto pq = oper_unpack(ipq);
            int p = pq.first;
            int q = pq.second;
            for(int irow=0; irow<rows; irow++){
               int isr = aindex[irow];
               auto sr = oper_unpack(isr);
               int s = sr.first;
               int r = sr.second;
               cmat(irow,icol) = int2e.get(p,q,s,r);
            } // irow
         } // icol
         return cmat;
      } 

   template <typename Tm>
      linalg::matrix<Tm> get_A2Pmat_su2(const std::vector<int>& aindex, 
            const std::vector<int>& pindex, 
            const integral::two_body<Tm>& int2e,
            const int ts){
         int rows = aindex.size();
         int cols = pindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ipq = pindex[icol];
            auto pq = oper_unpack(ipq);
            int p2 = pq.first, kp = p2/2;
            int q2 = pq.second, kq = q2/2;
            for(int irow=0; irow<rows; irow++){
               int isr = aindex[irow];
               auto sr = oper_unpack(isr);
               int s2 = sr.first, ks = s2/2;
               int r2 = sr.second, kr = r2/2;
               cmat(irow,icol) = get_xint2e_su2(int2e,ts,kp,kq,ks,kr);
            } // irow
         } // icol
         return cmat;
      } 

   template <typename Tm>
      linalg::matrix<Tm> get_B2Qmat(const std::vector<int>& bindex, 
            const std::vector<int>& qindex, 
            const integral::two_body<Tm>& int2e,
            const bool swap_qr){
         int rows = bindex.size();
         int cols = qindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ips = qindex[icol];
            auto ps = oper_unpack(ips);
            int p = ps.first;
            int s = ps.second;
            for(int irow=0; irow<rows; irow++){
               int iqr = bindex[irow];
               auto qr = oper_unpack(iqr);
               int q = qr.first;
               int r = qr.second;
               double wqr = (q==r)? 0.5 : 1.0;
               if(!swap_qr){
                  cmat(irow,icol) = wqr*int2e.get(p,q,s,r);
               }else{
                  cmat(irow,icol) = wqr*int2e.get(p,r,s,q);
               }
            } // irow
         } // icol
         return cmat;
      }

   template <typename Tm>
      linalg::matrix<Tm> get_B2Qmat_su2(const std::vector<int>& bindex, 
            const std::vector<int>& qindex, 
            const integral::two_body<Tm>& int2e,
            const int ts, 
            const bool swap_qr){
         int rows = bindex.size();
         int cols = qindex.size();
         linalg::matrix<Tm> cmat(rows,cols);
         for(int icol=0; icol<cols; icol++){
            int ips = qindex[icol];
            auto ps = oper_unpack(ips);
            int p2 = ps.first, kp = p2/2;
            int s2 = ps.second, ks = s2/2;
            for(int irow=0; irow<rows; irow++){
               int iqr = bindex[irow];
               auto qr = oper_unpack(iqr);
               int q2 = qr.first, kq = q2/2;
               int r2 = qr.second, kr = r2/2;
               double wqr = (kq==kr)? 0.5 : 1.0;
               if(!swap_qr){
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kq,ks,kr);
               }else{
                  if(ts == 2) wqr = -wqr; // (-1)^k in my note for Qps^k
                  cmat(irow,icol) = wqr*get_vint2e_su2(int2e,ts,kp,kr,ks,kq);
               }
            } // irow
         } // icol
         return cmat;
      }

} // ctns

#endif
