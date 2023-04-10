#ifndef SWEEP_TWODOT_DIAG_H
#define SWEEP_TWODOT_DIAG_H

#include "oper_dict.h"

namespace ctns{

   const bool debug_twodot_diag = false;
   extern const bool debug_twodot_diag;

   template <typename Tm>
      void twodot_diag(const oper_dictmap<Tm>& qops_dict,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1){
         const auto& lqops  = qops_dict.at("l");
         const auto& rqops  = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         if(rank == 0 && debug_twodot_diag){
            std::cout << "ctns::twodot_diag ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }

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

   template <typename Tm>
      void twodot_diags(const oper_dictmap<Tm>& qops_dict,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank,
            const bool ifdist1){
         const auto& lqops  = qops_dict.at("l");
         const auto& rqops  = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         if(rank == 0 && debug_twodot_diag){
            std::cout << "ctns::twodot_diag ifkr=" << lqops.ifkr 
               << " size=" << size << std::endl;
         }

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
         twodot_diag_BQs("lc1" ,  lqops, c1qops, wf, diag, size, rank);
         twodot_diag_BQs("lc2" ,  lqops, c2qops, wf, diag, size, rank);
         twodot_diag_BQs("lr"  ,  lqops,  rqops, wf, diag, size, rank);
         twodot_diag_BQs("c1c2", c1qops, c2qops, wf, diag, size, rank);
         twodot_diag_BQs("c1r" , c1qops,  rqops, wf, diag, size, rank);
         twodot_diag_BQs("c2r" , c2qops,  rqops, wf, diag, size, rank);
      }

   template <typename Tm>
      void twodot_diag_BQs(const std::string superblock,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank);
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diag_BQ superblock=" << superblock
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }

         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
            const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
            if(superblock == "lc1"){ 
               twodot_diag_OlOc1(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOc1(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "lc2"){ 
               twodot_diag_OlOc2(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOc2(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "lr"){
               twodot_diag_OlOr(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c1c2"){
               twodot_diag_Oc1Oc2(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc1Oc2(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c1r"){
               twodot_diag_Oc1Or(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc1Or(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c2r"){
               twodot_diag_Oc2Or(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc2Or(wt, O1.K(0), O2.K(0), wf, diag);
            }
         } // index

      }


   // H[loc] 
   template <typename Tm>
      void twodot_diag_local(const oper_dict<Tm>& lqops,
            const oper_dict<Tm>& rqops,
            const oper_dict<Tm>& c1qops,
            const oper_dict<Tm>& c2qops,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diag_local" << std::endl;
         }
         const auto& Hl  = lqops('H').at(0);
         const auto& Hr  = rqops('H').at(0);
         const auto& Hc1 = c1qops('H').at(0);
         const auto& Hc2 = c2qops('H').at(0);
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            const auto blkl = Hl(br,br);
            const auto blkr = Hr(bc,bc);
            const auto blkc1 = Hc1(bm,bm);
            const auto blkc2 = Hc2(bv,bv);
            size_t ircmv = wf.info._offset[idx]-1;
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
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const stensor4<Tm>& wf,
            double* diag,
            const int size,
            const int rank){
         const bool ifkr = qops1.ifkr;
         const bool ifNC = qops1.cindex.size() <= qops2.cindex.size();
         char BQ1 = ifNC? 'B' : 'Q';
         char BQ2 = ifNC? 'Q' : 'B';
         const auto& cindex = ifNC? qops1.cindex : qops2.cindex;
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank);
         if(rank == 0 && debug_twodot_diag){ 
            std::cout << "twodot_diag_BQ superblock=" << superblock
               << " ifNC=" << ifNC << " " << BQ1 << BQ2 
               << " size=" << bindex_dist.size() 
               << std::endl;
         }

         // B^L*Q^R or Q^L*B^R 
#ifdef _OPENMP

         size_t ndim = wf.size();
#pragma omp parallel
         {
            double* di = new double[ndim];
            memset(di, 0, ndim*sizeof(double));
#pragma omp for schedule(dynamic) nowait
            for(const auto& index : bindex_dist){
               const auto& O1 = qops1(BQ1).at(index);
               const auto& O2 = qops2(BQ2).at(index);
               if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
               const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
               if(superblock == "lc1"){ 
                  twodot_diag_OlOc1(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_OlOc1(wt, O1.K(0), O2.K(0), wf, di);
               }else if(superblock == "lc2"){ 
                  twodot_diag_OlOc2(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_OlOc2(wt, O1.K(0), O2.K(0), wf, di);
               }else if(superblock == "lr"){
                  twodot_diag_OlOr(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, di);
               }else if(superblock == "c1c2"){
                  twodot_diag_Oc1Oc2(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_Oc1Oc2(wt, O1.K(0), O2.K(0), wf, di);
               }else if(superblock == "c1r"){
                  twodot_diag_Oc1Or(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_Oc1Or(wt, O1.K(0), O2.K(0), wf, di);
               }else if(superblock == "c2r"){
                  twodot_diag_Oc2Or(wt, O1, O2, wf, di);
                  if(ifkr) twodot_diag_Oc2Or(wt, O1.K(0), O2.K(0), wf, di);
               }
            } // index
#pragma omp critical
            linalg::xaxpy(ndim, 1.0, di, diag);
            delete[] di;
         }

#else

         for(const auto& index : bindex_dist){
            const auto& O1 = qops1(BQ1).at(index);
            const auto& O2 = qops2(BQ2).at(index);
            if(O1.info.sym.is_nonzero()) continue; // screening for <l|B/Q^l_{pq}|l>
            const double wt = ifkr? 2.0*wfacBQ(index) : 2.0*wfac(index); // 2.0 from B^H*Q^H
            if(superblock == "lc1"){ 
               twodot_diag_OlOc1(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOc1(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "lc2"){ 
               twodot_diag_OlOc2(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOc2(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "lr"){
               twodot_diag_OlOr(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_OlOr(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c1c2"){
               twodot_diag_Oc1Oc2(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc1Oc2(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c1r"){
               twodot_diag_Oc1Or(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc1Or(wt, O1.K(0), O2.K(0), wf, diag);
            }else if(superblock == "c2r"){
               twodot_diag_Oc2Or(wt, O1, O2, wf, diag);
               if(ifkr) twodot_diag_Oc2Or(wt, O1.K(0), O2.K(0), wf, diag);
            }
         } // index

#endif
      }

   // Ol*Oc1
   template <typename Tm>
      void twodot_diag_OlOc1(const double wt,
            const stensor2<Tm>& Ol,
            const stensor2<Tm>& Oc1,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Ol*Oc1
            const auto blkl  = Ol(br,br); 
            const auto blkc1 = Oc1(bm,bm);
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkl(ir,ir)*blkc1(im,im));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Ol*Oc2
   template <typename Tm>
      void twodot_diag_OlOc2(const double wt,
            const stensor2<Tm>& Ol,
            const stensor2<Tm>& Oc2,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Ol*Oc2
            const auto blkl  = Ol(br,br); 
            const auto blkc2 = Oc2(bv,bv); 
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkl(ir,ir)*blkc2(iv,iv));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Ol*Or
   template <typename Tm>
      void twodot_diag_OlOr(const double wt,
            const stensor2<Tm>& Ol,
            const stensor2<Tm>& Or,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Ol*Or
            const auto blkl = Ol(br,br); 
            const auto blkr = Or(bc,bc); 
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkl(ir,ir)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc1*Oc2
   template <typename Tm>
      void twodot_diag_Oc1Oc2(const double wt,
            const stensor2<Tm>& Oc1,
            const stensor2<Tm>& Oc2,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Oc1*Oc2
            const auto blkc1 = Oc1(bm,bm); 
            const auto blkc2 = Oc2(bv,bv); 
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkc1(im,im)*blkc2(iv,iv));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc1*Or
   template <typename Tm>
      void twodot_diag_Oc1Or(const double wt,
            const stensor2<Tm>& Oc1,
            const stensor2<Tm>& Or,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Oc1*Or
            const auto blkc1 = Oc1(bm,bm); 
            const auto blkr  = Or(bc,bc); 
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkc1(im,im)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

   // Oc2*Or
   template <typename Tm>
      void twodot_diag_Oc2Or(const double wt,
            const stensor2<Tm>& Oc2,
            const stensor2<Tm>& Or,
            const stensor4<Tm>& wf,
            double* diag){
         const size_t ndim = wf.size();
#pragma omp parallel for reduction(+:diag[:ndim])
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            int br, bc, bm, bv;
            wf.info._addr_unpack(idx, br, bc, bm, bv);
            auto blk = wf(br,bc,bm,bv);
            int rdim = blk.dim0;
            int cdim = blk.dim1;
            int mdim = blk.dim2;
            int vdim = blk.dim3;
            // Oc2*Or
            const auto blkc2 = Oc2(bv,bv); 
            const auto blkr  = Or(bc,bc); 
            size_t ircmv = wf.info._offset[idx]-1;  
            for(int iv=0; iv<vdim; iv++){
               for(int im=0; im<mdim; im++){
                  for(int ic=0; ic<cdim; ic++){
                     for(int ir=0; ir<rdim; ir++){
                        diag[ircmv] += wt*std::real(blkc2(iv,iv)*blkr(ic,ic));
                        ircmv++;
                     } // ir
                  } // ic
               } // im
            } // iv
         } // i
      }

} // ctns

#endif
