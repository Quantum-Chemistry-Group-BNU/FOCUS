#ifndef SWEEP_ONEDOT_DIAG_SU2_H
#define SWEEP_ONEDOT_DIAG_SU2_H

#include "../oper_dict.h"

namespace ctns{

   const double thresh_diag_angular = 1.e-14;
   extern const double thresh_diag_angular;

   const bool debug_onedot_diag_su2 = false;
   extern const bool debug_onedot_diag_su2;

   template <typename Tm>
      void onedot_diag(const opersu2_dictmap<Tm>& qops_dict,
            const stensor3su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1){
         const auto& lqops = qops_dict.at("l"); 
         const auto& rqops = qops_dict.at("r"); 
         const auto& cqops = qops_dict.at("c"); 
         if(rank == 0 && debug_onedot_diag_su2){
            std::cout << "ctns::onedot_diag(su2) ifkr=" << lqops.ifkr 
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
      void onedot_diag_local(const opersu2_dict<Tm>& lqops,
            const opersu2_dict<Tm>& rqops,
            const opersu2_dict<Tm>& cqops,
            const stensor3su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         if(rank == 0 && debug_onedot_diag_su2){ 
            std::cout << "onedot_diag_local" << std::endl;
         }
         const auto& Hl = lqops('H').at(0);
         const auto& Hr = rqops('H').at(0);
         const auto& Hc = cqops('H').at(0);
         int br, bc, bm, tsi;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            auto key = wf.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            const auto blkl = Hl(br,br);
            const auto blkr = Hr(bc,bc);
            const auto blkc = Hc(bm,bm);
            size_t ircm = wf.info.get_offset(br,bc,bm,tsi)-1;
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
            const opersu2_dict<Tm>& qops1,
            const opersu2_dict<Tm>& qops2,
            const stensor3su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const size_t csize1 = qops1.cindex.size();
         const size_t csize2 = qops2.cindex.size();
         const bool ifNC = determine_NCorCN_BQ(qops1.oplist, qops2.oplist, csize1, csize2);
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         if(rank == 0 && debug_onedot_diag_su2){
            std::cout << "onedot_diag_BQ superblock=" << superblock 
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }

         // B^L*Q^R or Q^L*B^R 
         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            assert(O1.info.sym.ne() == 0);
            // determine spin rank
            auto pq = oper_unpack(index);
            int p = pq.first, kp = p/2, sp = p%2;
            int q = pq.second, kq = q/2, sq = q%2;
            int ts = (sp!=sq)? 2 : 0;
            double fac = (kp==kq)? 0.5 : 1.0;
            double wt = ((ts==0)? 1.0 : -std::sqrt(3.0))*fac*2.0; // 2.0 from B^H*Q^H
            if(superblock == "lc"){ 
               onedot_diag_OlOc(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "cr"){
               onedot_diag_OcOr(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "lr"){
               onedot_diag_OlOr(ts, wt, O1, O2, wf, diag);
            }
         } // index
      }

   template <typename Tm>
      double get_onedot_diag_su2info(const int i,
            const stensor3su2<Tm>& wf,
            int& br,
            int& bc,
            int& bm,
            int& tsi,
            const int tsOl,
            const int tsOc,
            const int tsOr,
            const int tsO1,
            const int tsO2,
            const int tsOtot,
            const double wt){
         int tstot = wf.info.sym.ts();
         auto key = wf.info._nnzaddr[i];
         br = std::get<0>(key);
         bc = std::get<1>(key);
         bm = std::get<2>(key);
         tsi = std::get<3>(key);
         int tsl = wf.info.qrow.get_sym(br).ts();
         int tsr = wf.info.qcol.get_sym(bc).ts();
         int tsc = wf.info.qmid.get_sym(bm).ts();
         // spin factor
         double fac;
         if(wf.info.couple == CRcouple){
            // l|cr: (<Slp|Ol|Sl>(<Scp|Oc|Sc><Srp|Or|Sr>)[ScrpScr])[Stot]
            int tscr = tsi;
            fac = wt*std::sqrt((tsl+1.0)*(tscr+1.0)*(tstot+1.0)*(tsOtot+1.0))*
               fock::wigner9j(tsl,tscr,tstot,tsl,tscr,tstot,tsO1,tsO2,tsOtot)*
               std::sqrt((tsc+1.0)*(tsr+1.0)*(tscr+1.0)*(tsO2+1.0))*
               fock::wigner9j(tsc,tsr,tscr,tsc,tsr,tscr,tsOc,tsOr,tsO2);
         }else{
            // lc|r: ((<Slp|Ol|Sl><Scp|Oc|Sc>)[SlcpSlc]<Srp|Or|Sr>))[Stot]
            int tslc = tsi;
            fac = wt*std::sqrt((tslc+1.0)*(tsr+1.0)*(tstot+1.0)*(tsOtot+1.0))*
               fock::wigner9j(tslc,tsr,tstot,tslc,tsr,tstot,tsO1,tsO2,tsOtot)*
               std::sqrt((tsl+1.0)*(tsc+1.0)*(tslc+1.0)*(tsO1+1.0))*
               fock::wigner9j(tsl,tsc,tslc,tsl,tsc,tslc,tsOl,tsOc,tsO1);
         }
         return fac;
      }

   // Ol*Oc*Ir
   template <typename Tm>
      void onedot_diag_OlOc(const int ts,
            const double wt,
            const stensor2su2<Tm>& Ol,
            const stensor2su2<Tm>& Oc,
            const stensor3su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = ts;
         int tsOc = ts;
         int tsOr = 0;
         int tsO1 = 0, tsO2 = 0, tsOtot = 0;
         if(wf.info.couple == CRcouple){
            // l|cr: Ol*(Oc*Ir)
            tsO1 = ts; tsO2 = ts;
         }else{
            // lc|r: (Ol*Oc)*Ir
            tsO1 = 0; tsO2 = 0;
         }
         int br, bc, bm, tsi;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                 tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            // Ol*Oc 
            const auto blkl = Ol(br,br);
            const auto blkc = Oc(bm,bm);
            assert(blkl.size() > 0 and blkc.size() > 0);
            size_t ircm = wf.info.get_offset(br,bc,bm,tsi)-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += fac*std::real(blkl(ir,ir)*blkc(im,im));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

   // Ol*Ic*Or
   template <typename Tm>
      void onedot_diag_OlOr(const int ts,
            const double wt,
            const stensor2su2<Tm>& Ol,
            const stensor2su2<Tm>& Or,
            const stensor3su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = ts;
         int tsOc = 0;
         int tsOr = ts;
         int tsO1 = 0, tsO2 = 0, tsOtot = 0;
         if(wf.info.couple == CRcouple){
            // l|cr: Ol*(Ic*Or)
            tsO1 = ts; tsO2 = ts;
         }else{
            // lc|r: (Ol*Ic)*Or
            tsO1 = ts; tsO2 = ts;
         }
         int br, bc, bm, tsi;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                 tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            // Ol*Or
            const auto blkl = Ol(br,br);
            const auto blkr = Or(bc,bc);
            assert(blkl.size() > 0 and blkr.size() > 0);
            size_t ircm = wf.info.get_offset(br,bc,bm,tsi)-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += fac*std::real(blkl(ir,ir)*blkr(ic,ic));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

   // Il*Oc*Or
   template <typename Tm>
      void onedot_diag_OcOr(const int ts,
            const double wt,
            const stensor2su2<Tm>& Oc,
            const stensor2su2<Tm>& Or,
            const stensor3su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = 0;
         int tsOc = ts;
         int tsOr = ts;
         int tsO1 = 0, tsO2 = 0, tsOtot = 0;
         if(wf.info.couple == CRcouple){
            // l|cr: Il*(Oc*Or)
            tsO1 = 0; tsO2 = 0;
         }else{
            // lc|r: (Il*Oc)*Or
            tsO1 = ts; tsO2 = ts;
         }
         int br, bc, bm, tsi;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_onedot_diag_su2info(i,wf,br,bc,bm,tsi,
                 tsOl,tsOc,tsOr,tsO1,tsO2,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            // Oc*Or
            const auto blkc = Oc(bm,bm);
            const auto blkr = Or(bc,bc);
            assert(blkc.size() > 0 and blkr.size() > 0);
            size_t ircm = wf.info.get_offset(br,bc,bm,tsi)-1;
            for(int im=0; im<mdim; im++){
               for(int ic=0; ic<cdim; ic++){
                  for(int ir=0; ir<rdim; ir++){
                     diag[ircm] += fac*std::real(blkc(im,im)*blkr(ic,ic));
                     ircm++;
                  } // ir
               } // ic
            } // im
         } // i
      }

} // ctns

#endif
