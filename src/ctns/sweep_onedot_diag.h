#ifndef SWEEP_ONEDOT_DIAG_H
#define SWEEP_ONEDOT_DIAG_H

#include "oper_dict.h"

namespace ctns{

   const bool debug_onedot_diag = false;
   extern const bool debug_onedot_diag;

   template <typename Tm>
      void onedot_diag(const oper_dictmap<Tm>& qops_dict,
            const stensor3<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1){
         const auto& lqops = qops_dict.at("l"); 
         const auto& rqops = qops_dict.at("r"); 
         const auto& cqops = qops_dict.at("c"); 
         if(rank == 0 && debug_onedot_diag){
            std::cout << "ctns::onedot_diag ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }

         // 0. constant term 
         size_t ndim = wf.size();
         memset(diag, 0, ndim*sizeof(double));

         // 1. local terms: <lcr|H|lcr> = Hll + Hcc + Hrr
         // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
         // whose contributions to Diag need to be taken into aacount.
         if(!ifdist1 || rank == 0){
            onedot_diag_local(lqops, rqops, cqops, wf, diag, size, rank);
         }

         // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
         //         B/Q^C
         //         |
         // B/Q^L---*---B/Q^R
         onedot_diag_BQ("lc", lqops, cqops, wf, diag, size, rank);
         onedot_diag_BQ("lr", lqops, rqops, wf, diag, size, rank);
         onedot_diag_BQ("cr", cqops, rqops, wf, diag, size, rank);
      }

   // H[loc] 
   template <typename Tm>
      void onedot_diag_local(const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& cqops,
            const stensor3<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         if(rank == 0 && debug_onedot_diag){ 
            std::cout << "onedot_diag_local" << std::endl;
         }
         const auto& Hl = lqops('H').at(0);
         const auto& Hr = rqops('H').at(0);
         const auto& Hc = cqops('H').at(0);
         int br, bc, bm;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx, br, bc, bm);
            auto blk = wf(br,bc,bm);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            const auto blkl = Hl(br,br);
            const auto blkr = Hr(bc,bc);
            const auto blkc = Hc(bm,bm);
            size_t ircm = wf.info._offset[idx]-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += std::real(blkl(ir,ir)) 
                        + std::real(blkr(ic,ic))
                        + std::real(blkc(im,im));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

   template <typename Tm>
      void onedot_diag_BQ(const std::string superblock,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor3<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         const int isym = qops1.isym;
         const bool ifkr = qops1.ifkr;
         const size_t csize1 = qops1.cindex.size();
         const size_t csize2 = qops2.cindex.size();
         const bool ifNC = determine_NCorCN_BQ(qops1.oplist, qops2.oplist, csize1, csize2);
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, isym, size, rank, qops1.sorb);
         if(rank == 0 && debug_onedot_diag){
            std::cout << "onedot_diag_BQ superblock=" << superblock 
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }
         // B^L*Q^R or Q^L*B^R 
         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
            const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
            if(superblock == "lc"){ 
               onedot_diag_OlOc(wt, O1, O2, wf, diag);
               if(ifkr) onedot_diag_OlOc(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "cr"){
               onedot_diag_OcOr(wt, O1, O2, wf, diag);
               if(ifkr) onedot_diag_OcOr(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "lr"){
               onedot_diag_OlOr(wt, O1, O2, wf, diag);
               if(ifkr) onedot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, diag);
            }
         } // index
      }

   // Ol*Oc*Ir
   template <typename Tm>
      void onedot_diag_OlOc(const double wt,
            const stensor2<Tm>& Ol,
            const stensor2<Tm>& Oc,
            const stensor3<Tm>& wf,
            double* diag){
         int br, bc, bm;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx, br, bc, bm);
            auto blk = wf(br,bc,bm);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            // Ol*Oc 
            const auto blkl = Ol(br,br);
            const auto blkc = Oc(bm,bm);
            size_t ircm = wf.info._offset[idx]-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += wt*std::real(blkl(ir,ir)*blkc(im,im));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

   // Ol*Ic*Or
   template <typename Tm>
      void onedot_diag_OlOr(const double wt,
            const stensor2<Tm>& Ol,
            const stensor2<Tm>& Or,
            const stensor3<Tm>& wf,
            double* diag){
         int br, bc, bm;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx, br, bc, bm);
            auto blk = wf(br,bc,bm);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            // Ol*Or
            const auto blkl = Ol(br,br);
            const auto blkr = Or(bc,bc);
            size_t ircm = wf.info._offset[idx]-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += wt*std::real(blkl(ir,ir)*blkr(ic,ic));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

   // Il*Oc*Or
   template <typename Tm>
      void onedot_diag_OcOr(const double wt,
            const stensor2<Tm>& Oc,
            const stensor2<Tm>& Or,
            const stensor3<Tm>& wf,
            double* diag){
         int br, bc, bm;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx, br, bc, bm);
            auto blk = wf(br,bc,bm);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            // Oc*Or
            const auto blkc = Oc(bm,bm);
            const auto blkr = Or(bc,bc);
            size_t ircm = wf.info._offset[idx]-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += wt*std::real(blkc(im,im)*blkr(ic,ic));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

} // ctns

#endif
