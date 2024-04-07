#ifndef SWEEP_TWODOT_DIAG_SU2_H
#define SWEEP_TWODOT_DIAG_SU2_H

#include "../oper_dict.h"

namespace ctns{

   const bool debug_twodot_diag_su2 = false;
   extern const bool debug_twodot_diag_su2;

   const double thresh_diag_angular = 1.e-14;
   extern const double thresh_diag_angular;

   template <typename Tm>
      void twodot_diag(const opersu2_dictmap<Tm>& qops_dict,
            const stensor4su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1){
         const auto& lqops  = qops_dict.at("l");
         const auto& rqops  = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         if(rank == 0 && debug_twodot_diag_su2){
            std::cout << "ctns::twodot_diag(su2) ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }

         // 0. cleanup 
         size_t ndim = wf.size();
         memset(diag, 0, ndim*sizeof(double));

         // 1. local terms: <lc1c2r|H|lc1c2r> = Hll + Hc1c1 + Hc2c2 + Hrr
         // NOTE: ifdist1=false, each node has nonzero H[l] and H[r],
         // whose contributions to Diag need to be taken into aacount.
         if(!ifdist1 || rank == 0){
            twodot_diag_local(lqops, rqops, c1qops, c2qops, wf, diag, size, rank);
         }

         // 2. density-density interactions: BQ terms where (p^+q)(r^+s) in two of l/c/r
         //        B/Q^C1 B/Q^C2
         //         |      |
         // B/Q^L---*------*---B/Q^R
         twodot_diag_BQ("lc1" ,  lqops, c1qops, wf, diag, size, rank);
         twodot_diag_BQ("lc2" ,  lqops, c2qops, wf, diag, size, rank);
         twodot_diag_BQ("lr"  ,  lqops,  rqops, wf, diag, size, rank);
         twodot_diag_BQ("c1c2", c1qops, c2qops, wf, diag, size, rank);
         twodot_diag_BQ("c1r" , c1qops,  rqops, wf, diag, size, rank);
         twodot_diag_BQ("c2r" , c2qops,  rqops, wf, diag, size, rank);

      }

   // H[loc] 
   template <typename Tm>
      void twodot_diag_local(const opersu2_dict<Tm>& lqops,
            const opersu2_dict<Tm>& rqops,
            const opersu2_dict<Tm>& c1qops,
            const opersu2_dict<Tm>& c2qops,
            const stensor4su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         if(rank == 0 && debug_twodot_diag_su2){ 
            std::cout << "twodot_diag_local" << std::endl;
         }
         const auto& Hl  = lqops('H').at(0);
         const auto& Hr  = rqops('H').at(0);
         const auto& Hc1 = c1qops('H').at(0);
         const auto& Hc2 = c2qops('H').at(0);
         int br, bc, bm, bv, tslc1, tsc2r, tstot;
         tstot = wf.info.sym.ts();
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            auto key = wf.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            bv = std::get<3>(key);
            tslc1 = std::get<4>(key);
            tsc2r = std::get<5>(key);
            int tsl  = wf.info.qrow.get_sym(br).ts();
            int tsr  = wf.info.qcol.get_sym(bc).ts();
            int tsc1 = wf.info.qmid.get_sym(bm).ts();
            int tsc2 = wf.info.qver.get_sym(bv).ts(); 
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            const auto blkl = Hl(br,br);
            const auto blkr = Hr(bc,bc);
            const auto blkc1 = Hc1(bm,bm);
            const auto blkc2 = Hc2(bv,bv);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += std::real(blkl(ir,ir))
                           + std::real(blkr(ic,ic))
                           + std::real(blkc1(im,im))
                           + std::real(blkc2(iv,iv));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   template <typename Tm>
      void twodot_diag_BQ(const std::string superblock,
            const opersu2_dict<Tm>& qops1,
            const opersu2_dict<Tm>& qops2,
            const stensor4su2<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, qops1.sorb);
         if(rank == 0 && debug_twodot_diag_su2){ 
            std::cout << "twodot_diag_BQ superblock=" << superblock
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
            if(superblock == "lc1"){ 
               twodot_diag_OlOc1(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "lc2"){ 
               twodot_diag_OlOc2(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "lr"){
               twodot_diag_OlOr(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "c1c2"){
               twodot_diag_Oc1Oc2(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "c1r"){
               twodot_diag_Oc1Or(ts, wt, O1, O2, wf, diag);
            }else if(superblock == "c2r"){
               twodot_diag_Oc2Or(ts, wt, O1, O2, wf, diag);
            }
         } // index
      }

   template <typename Tm>
      double get_twodot_diag_su2info(const int i,
            const stensor4su2<Tm>& wf,
            int& br,
            int& bc,
            int& bm,
            int& bv,
            int& tslc1,
            int& tsc2r,
            const int tsOl,
            const int tsOc1,
            const int tsOlc1,
            const int tsOc2,
            const int tsOr,
            const int tsOc2r,
            const int tsOtot,
            const double wt){
         int tstot = wf.info.sym.ts();
         auto key = wf.info._nnzaddr[i];
         br = std::get<0>(key);
         bc = std::get<1>(key);
         bm = std::get<2>(key);
         bv = std::get<3>(key);
         tslc1 = std::get<4>(key);
         tsc2r = std::get<5>(key);
         int tsl  = wf.info.qrow.get_sym(br).ts();
         int tsr  = wf.info.qcol.get_sym(bc).ts();
         int tsc1 = wf.info.qmid.get_sym(bm).ts();
         int tsc2 = wf.info.qver.get_sym(bv).ts(); 
         // spin factor
         // ((<Slp|Ol|Sl><Sc1p|Oc1|Sc1>)[Slc1p,Slc1](<Sc2p|Oc2|Sc2><Srp|Or|Sr>)[Sc2rpSc2r])[Stot]
         double fac = wt*std::sqrt((tslc1+1.0)*(tsc2r+1.0)*(tstot+1.0)*(tsOtot+1.0))*
            fock::wigner9j(tslc1,tsc2r,tstot,tslc1,tsc2r,tstot,tsOlc1,tsOc2r,tsOtot)*
            std::sqrt((tsl+1.0)*(tsc1+1.0)*(tslc1+1.0)*(tsOlc1+1.0))*
            fock::wigner9j(tsl,tsc1,tslc1,tsl,tsc1,tslc1,tsOl,tsOc1,tsOlc1)*
            std::sqrt((tsc2+1.0)*(tsr+1.0)*(tsc2r+1.0)*(tsOc2r+1.0))*
            fock::wigner9j(tsc2,tsr,tsc2r,tsc2,tsr,tsc2r,tsOc2,tsOr,tsOc2r);
         return fac;
      }

   // Ol*Oc1
   template <typename Tm>
      void twodot_diag_OlOc1(const int ts,
            const double wt,
            const stensor2su2<Tm>& Ol,
            const stensor2su2<Tm>& Oc1,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = ts;
         int tsOc1 = ts;
         int tsOlc1 = 0;
         int tsOc2 = 0;
         int tsOr = 0;
         int tsOc2r = 0;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Ol*Oc1
            const auto blkl  = Ol(br,br);
            const auto blkc1 = Oc1(bm,bm);
            assert(blkl.size() > 0 and blkc1.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkl(ir,ir)*blkc1(im,im));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Ol*Oc2
   template <typename Tm>
      void twodot_diag_OlOc2(const int ts,
            const double wt,
            const stensor2su2<Tm>& Ol,
            const stensor2su2<Tm>& Oc2,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = ts;
         int tsOc1 = 0;
         int tsOlc1 = ts;
         int tsOc2 = ts;
         int tsOr = 0;
         int tsOc2r = ts;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Ol*Oc2
            const auto blkl  = Ol(br,br); 
            const auto blkc2 = Oc2(bv,bv);
            assert(blkl.size() > 0 and blkc2.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkl(ir,ir)*blkc2(iv,iv));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Ol*Or
   template <typename Tm>
      void twodot_diag_OlOr(const int ts,
            const double wt,
            const stensor2su2<Tm>& Ol,
            const stensor2su2<Tm>& Or,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = ts;
         int tsOc1 = 0;
         int tsOlc1 = ts;
         int tsOc2 = 0;
         int tsOr = ts;
         int tsOc2r = ts;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Ol*Or
            const auto blkl = Ol(br,br); 
            const auto blkr = Or(bc,bc); 
            assert(blkl.size() > 0 and blkr.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkl(ir,ir)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc1*Oc2
   template <typename Tm>
      void twodot_diag_Oc1Oc2(const int ts,
            const double wt,
            const stensor2su2<Tm>& Oc1,
            const stensor2su2<Tm>& Oc2,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = 0;
         int tsOc1 = ts;
         int tsOlc1 = ts;
         int tsOc2 = ts;
         int tsOr = 0;
         int tsOc2r = ts;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Oc1*Oc2
            const auto blkc1 = Oc1(bm,bm); 
            const auto blkc2 = Oc2(bv,bv); 
            assert(blkc1.size() > 0 and blkc2.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkc1(im,im)*blkc2(iv,iv));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc1*Or
   template <typename Tm>
      void twodot_diag_Oc1Or(const int ts,
            const double wt,
            const stensor2su2<Tm>& Oc1,
            const stensor2su2<Tm>& Or,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = 0;
         int tsOc1 = ts;
         int tsOlc1 = ts;
         int tsOc2 = 0;
         int tsOr = ts;
         int tsOc2r = ts;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Oc1*Or
            const auto blkc1 = Oc1(bm,bm); 
            const auto blkr  = Or(bc,bc); 
            assert(blkc1.size() > 0 and blkr.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkc1(im,im)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc2*Or
   template <typename Tm>
      void twodot_diag_Oc2Or(const int ts,
            const double wt,
            const stensor2su2<Tm>& Oc2,
            const stensor2su2<Tm>& Or,
            const stensor4su2<Tm>& wf,
            double* diag){
         // rank of operators
         int tsOl = 0;
         int tsOc1 = 0;
         int tsOlc1 = 0;
         int tsOc2 = ts;
         int tsOr = ts;
         int tsOc2r = 0;
         int tsOtot = 0;
         int br, bc, bm, bv, tslc1, tsc2r;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            double fac = get_twodot_diag_su2info(i,wf,br,bc,bm,bv,tslc1,tsc2r,
                  tsOl,tsOc1,tsOlc1,tsOc2,tsOr,tsOc2r,tsOtot,wt);
            if(std::abs(fac) < thresh_diag_angular) continue;
            int rdim = wf.info.qrow.get_dim(br);
            int cdim = wf.info.qcol.get_dim(bc);
            int mdim = wf.info.qmid.get_dim(bm);
            int vdim = wf.info.qver.get_dim(bv);
            // Oc2*Or
            const auto blkc2 = Oc2(bv,bv); 
            const auto blkr  = Or(bc,bc); 
            assert(blkc2.size() > 0 and blkr.size() > 0);
            size_t ircmv = wf.info.get_offset(br,bc,bm,bv,tslc1,tsc2r)-1;
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += fac*std::real(blkc2(iv,iv)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

} // ctns

#endif
