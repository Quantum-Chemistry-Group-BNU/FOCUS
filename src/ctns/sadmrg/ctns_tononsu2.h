#ifndef CTNS_TONONSU2_H
#define CTNS_TONONSU2_H

#include "ctns_tosu2_qbond3.h"

namespace ctns{

   inline std::pair<qbond,std::map<qsym3,std::pair<int,int>>> qbond_su2expand(const qbond& qs){
      std::map<qsym,std::vector<int>> qmap;
      for(int i=0; i<qs.size(); i++){
         auto sym = qs.dims[i].first;
         auto dim = qs.dims[i].second;
         int ne = sym.ne();
         int ts = sym.ts();
         for(int tm=-ts; tm<=ts; tm+=2){
            qmap[qsym(2,ne,tm)].push_back(i);
         }
      }
      qbond qs2;
      std::map<qsym3,std::pair<int,int>> offmap;
      int b = 0;
      for(const auto& pr : qmap){
         auto sym0 = pr.first;
         auto indices = pr.second;
         int ne = sym0.ne();
         int tm = sym0.tm();
         int dimtot = 0;
         for(int i=0; i<indices.size(); i++){
            int idx = indices[i];
            auto sym = qs.get_sym(idx);
            auto dim = qs.get_dim(idx);
            assert(sym.ne() == ne);
            int ts = sym.ts();
            offmap[qsym3(ne,ts,tm)] = std::make_pair(b,dimtot);
            dimtot += dim;
         }
         qs2.dims.push_back(std::make_pair(sym0,dimtot));
         b += 1;
      }
      return std::make_pair(qs2,offmap);
   }
      
   template <typename Tm>
      void qtensor3_tononsu2(const stensor3su2<Tm>& site,
            stensor3<Tm>& site2){
         assert(site.info.dir == dir_RCF);
         assert(site.info.couple == CRcouple);
         const auto& qrow = site.info.qrow;
         const auto& qcol = site.info.qcol;
         const auto& qmid = site.info.qmid;
         auto qexpandr = qbond_su2expand(qrow);
         auto qexpandc = qbond_su2expand(qcol);
         auto qexpandm = qbond_su2expand(qmid);
         const auto& qrow2 = qexpandr.first;
         const auto& qcol2 = qexpandc.first;
         const auto& qmid2 = qexpandm.first;
         const auto& offmapr = qexpandr.second;
         const auto& offmapc = qexpandc.second;
         const auto& offmapm = qexpandm.second;
         site2.init(qsym(2,0,0),qrow2,qcol2,qmid2,dir_RCF);
         for(int i=0; i<site.info._nnzaddr.size(); i++){
            auto key = site.info._nnzaddr[i];
            int br = std::get<0>(key);
            int bc = std::get<1>(key);
            int bm = std::get<2>(key);
            int tsi = std::get<3>(key);
            auto qr = site.info.qrow.get_sym(br); // left
            auto qc = site.info.qcol.get_sym(bc); // right
            auto qm = site.info.qmid.get_sym(bm); // middle
            int tsr = qr.ts(); 
            int tsc = qc.ts();
            int tsm = qm.ts();
            assert(tsi == tsr); // CRcouple
            const auto blk = site(br,bc,bm,tsi);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            for(int tmm=-tsm; tmm<=tsm; tmm+=2){
               auto infom = offmapm.at(qsym3(qm.ne(),qm.ts(),tmm));
               int bm2 = infom.first;
               int offm = infom.second;
               for(int tmc=-tsc; tmc<=tsc; tmc+=2){
                  auto infoc = offmapc.at(qsym3(qc.ne(),qc.ts(),tmc));
                  int bc2 = infoc.first;
                  int offc = infoc.second;
                  for(int tmr=-tsr; tmr<=tsr; tmr+=2){
                     auto infor = offmapr.at(qsym3(qr.ne(),qr.ts(),tmr));
                     int br2 = infor.first;
                     int offr = infor.second;
                     // spin factor <SkMkS[rk]M[rk]|S[rk-1]M[rk-1]>
                     double cgfac = fock::cgcoeff(tsm,tsc,tsr,tmm,tmc,tmr);
                     // block
                     auto blk2 = site2(br2,bc2,bm2);
                     for(int im=0; im<mdim; im++){
                        int im2 = offm + im;
                        for(int ic=0; ic<cdim; ic++){
                           int ic2 = offc + ic;
                           for(int ir=0; ir<rdim; ir++){
                              int ir2 = offr + ir;
                              blk2(ir2,ic2,im2) = cgfac*blk(ir,ic,im);
                           } // ir
                        } // ic
                     } // im
                  } // tmr
               } // tmc
            } // tmm
         } // i
      }

   template <typename Tm>
      void qtensor2_tononsu2(const stensor2su2<Tm>& rwfun,
            stensor2<Tm>& rwfun2){


      }

   // convert SU2 mps to non-SU2 mps by a simple expansion
   template <typename Tm>
      void rcanon_tononsu2(const comb<qkind::qNS,Tm>& icomb,
            comb<qkind::qNSz,Tm>& icomb_NSz){
         const bool debug = true;
         std::cout << "\nctns::rcanon_tononsu2" << std::endl;
         icomb_NSz.topo = icomb.topo;
         // sites
         int nphysical = icomb.get_nphysical();
         icomb_NSz.sites.resize(nphysical);
         for(int i=0; i<nphysical; i++){
            qtensor3_tononsu2(icomb.sites[i],icomb_NSz.sites[i]); 
         }
         // rwfuns
         int nroots = icomb.get_nroots();
         icomb_NSz.rwfuns.resize(nroots);
         for(int i=0; i<nroots; i++){
            qtensor2_tononsu2(icomb.rwfuns[i],icomb_NSz.rwfuns[i]);
         }
      }

} // ctns

#endif
