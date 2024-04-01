#ifndef RESHAPE_QT4_QT3_SU2_H
#define RESHAPE_QT4_QT3_SU2_H

#include "stensor3su2.h"
#include "stensor4su2.h"

namespace ctns{

   // --- two-dot wavefunction: qt4[l,r,c1,c2] <-> qt3 ---
   // merge_lc1: wf4[l,r,c1,c2] => wf3[lc1,r,c2]
   // merge_c2r: wf4[l,r,c1,c2] => wf3[l,c2r,c1]

   // psi4[l,r,c1,c2] -> psi3[lc1,r,c2]
   template <typename Tm>
      stensor3su2<Tm> merge_qt4_qt3_lc1(const stensor4su2<Tm>& qt4,
            const qbond& qlc1, 
            const qdpt& dpt){
         assert(qt4.info.couple == LC1andC2Rcouple);
         const auto& sym = qt4.info.sym;
         const auto& qver = qt4.info.qver;
         const auto& qcol = qt4.info.qcol;
         direction3 dir = {1,1,1};
         stensor3su2<Tm> qt3(sym, qlc1, qcol, qver, dir, CRcouple);
         // loop over qt3
         int blc1, bc, bv, tsc2r;
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            blc1  = std::get<0>(key);
            bc    = std::get<1>(key);
            bv    = std::get<2>(key);
            tsc2r = std::get<3>(key);
            auto blk3 = qt3(blc1,bc,bv,tsc2r);
            // loop over compatible l*c1 = lc1
            auto qsym_lc1 = qlc1.get_sym(blc1);
            int tslc1 = qsym_lc1.ts(); 
            for(const auto& p12 : dpt.at(qsym_lc1)){
               int br = std::get<0>(p12);
               int bm = std::get<1>(p12);
               int ioff = std::get<2>(p12);
               const auto blk4 = qt4(br,bc,bm,bv,tslc1,tsc2r);
               if(blk4.empty()) continue;
               int rdim = blk4.dim0;
               int cdim = blk4.dim1;
               int mdim = blk4.dim2; 
               int vdim = blk4.dim3; 
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           // psi4[l,r,c1,c2] <-> psi3[l(fast)c1,r,c2]
                           int irm = ioff+im*rdim+ir; 
                           blk3(irm,ic,iv) = blk4(ir,ic,im,iv); // internally c1[fast]c2
                        } // ir
                     } // ic
                  } // im
               } // iv
            } // p12
         } // i
         return qt3;
      }
   // psi4[l,r,c1,c2] <- psi3[lc1,r,c2]
   template <typename Tm>
      stensor4su2<Tm> split_qt4_qt3_lc1(const stensor3su2<Tm>& qt3,
            const qbond& qlx,
            const qbond& qc1, 
            const qdpt& dpt){
         assert(qt3.info.couple == CRcouple);
         const auto& sym = qt3.info.sym;
         const auto& qlc1 = qt3.info.qrow;
         const auto& qver = qt3.info.qmid;
         const auto& qcol = qt3.info.qcol;
         stensor4su2<Tm> qt4(sym, qlx, qcol, qc1, qver, LC1andC2Rcouple);
         // loop over qt3
         int blc1, bc, bv, tsc2r;
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            blc1  = std::get<0>(key);
            bc    = std::get<1>(key);
            bv    = std::get<2>(key);
            tsc2r = std::get<3>(key);
            const auto blk3 = qt3(blc1,bc,bv,tsc2r);
            // loop over compatible l*c = lc
            auto qsym_lc1 = qlc1.get_sym(blc1);
            int tslc1 = qsym_lc1.ts();
            for(const auto& p12 : dpt.at(qsym_lc1)){
               int br = std::get<0>(p12);
               int bm = std::get<1>(p12);
               int ioff = std::get<2>(p12);
               auto blk4 = qt4(br,bc,bm,bv,tslc1,tsc2r);
               if(blk4.empty()) continue;
               int rdim = blk4.dim0;
               int cdim = blk4.dim1;
               int mdim = blk4.dim2; 
               int vdim = blk4.dim3; 
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        for(int ir=0; ir<rdim; ir++){
                           // psi4[l,r,c1,c2] <-> psi3[lc1,r,c2]
                           int irm = ioff+im*rdim+ir; // store (ir,im) 
                           blk4(ir,ic,im,iv) = blk3(irm,ic,iv); // internally c1[fast]c2
                        } // ir
                     } // ic
                  } // im
               } // iv
            } // p12
         } // i
         return qt4;
      }

   // psi4[l,r,c1,c2] -> psi3[l,c2r,c1]
   template <typename Tm>
      stensor3su2<Tm> merge_qt4_qt3_c2r(const stensor4su2<Tm>& qt4,
            const qbond& qc2r, 
            const qdpt& dpt){
         assert(qt4.info.couple == LC1andC2Rcouple);
         const auto& sym = qt4.info.sym;
         const auto& qrow = qt4.info.qrow; 
         const auto& qmid = qt4.info.qmid;
         direction3 dir = {1,1,1};
         stensor3su2<Tm> qt3(sym, qrow, qc2r, qmid, dir, LCcouple);
         // loop over qt3
         int br, bc2r, bm, tslc1;
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            br    = std::get<0>(key);
            bc2r  = std::get<1>(key);
            bm    = std::get<2>(key);
            tslc1 = std::get<3>(key);
            auto blk3 = qt3(br,bc2r,bm,tslc1);
            // loop over compatible c2*r = c2r
            auto qsym_c2r = qc2r.get_sym(bc2r);
            int tsc2r = qsym_c2r.ts();
            for(const auto& p12 : dpt.at(qsym_c2r)){
               int bv = std::get<0>(p12);
               int bc = std::get<1>(p12);
               int ioff = std::get<2>(p12);
               const auto blk4 = qt4(br,bc,bm,bv,tslc1,tsc2r);
               if(blk4.empty()) continue;
               int rdim = blk4.dim0;
               int cdim = blk4.dim1;
               int mdim = blk4.dim2; 
               int vdim = blk4.dim3; 
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        // psi4[l,r,c1,c2] <-> psi3[l,c2r,c1]
                        int ivc = ioff+ic*vdim+iv;
                        for(int ir=0; ir<rdim; ir++){
                           blk3(ir,ivc,im) = blk4(ir,ic,im,iv);
                        } // ir
                     } // ic
                  } // im
               } // iv
            } // p12
         } // i
         return qt3;
      }
   // psi4[l,r,c1,c2] <- psi3[l,c2r,c1]
   template <typename Tm>
      stensor4su2<Tm> split_qt4_qt3_c2r(const stensor3su2<Tm>& qt3,
            const qbond& qc2,
            const qbond& qrx, 
            const qdpt& dpt){
         assert(qt3.info.couple == LCcouple);
         const auto& sym = qt3.info.sym;
         const auto& qrow = qt3.info.qrow; 
         const auto& qmid = qt3.info.qmid;
         const auto& qc2r = qt3.info.qcol;
         stensor4su2<Tm> qt4(sym, qrow, qrx, qmid, qc2, LC1andC2Rcouple);
         // loop over qt3
         int br, bc2r, bm, tslc1;
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            br    = std::get<0>(key);
            bc2r  = std::get<1>(key);
            bm    = std::get<2>(key);
            tslc1 = std::get<3>(key);
            const auto blk3 = qt3(br,bc2r,bm,tslc1);
            // loop over compatible c2*r = c2r
            auto qsym_c2r = qc2r.get_sym(bc2r);
            int tsc2r = qsym_c2r.ts();
            for(const auto& p12 : dpt.at(qsym_c2r)){
               int bv = std::get<0>(p12);
               int bc = std::get<1>(p12);
               int ioff = std::get<2>(p12);
               auto blk4 = qt4(br,bc,bm,bv,tslc1,tsc2r);
               if(blk4.empty()) continue;
               int rdim = blk4.dim0;
               int cdim = blk4.dim1;
               int mdim = blk4.dim2; 
               int vdim = blk4.dim3; 
               for(int iv=0; iv<vdim; iv++){
                  for(int im=0; im<mdim; im++){
                     for(int ic=0; ic<cdim; ic++){
                        // psi4[l,r,c1,c2] <-> psi3[l,c2r,c1]
                        int ivc = ioff+ic*vdim+iv;
                        for(int ir=0; ir<rdim; ir++){
                           blk4(ir,ic,im,iv) = blk3(ir,ivc,im);
                        } // ir
                     } // ic
                  } // im
               } // iv
            } // p12
         } // i
         return qt4;
      }

} // ctns

#endif
