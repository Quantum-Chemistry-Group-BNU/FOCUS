#ifndef CONTRACT_QT3_QT2_SU2_H
#define CONTRACT_QT3_QT2_SU2_H

#include "stensor2su2.h"
#include "stensor3su2.h"

namespace ctns{

   // --- contract_qt3_qt2 ---
   template <typename Tm>
      stensor3su2<Tm> contract_qt3_qt2(const std::string cpos,
            const stensor3su2<Tm>& qt3a, 
            const stensor2su2<Tm>& qt2,
            const bool iftrans=false){
   
         // symmetry cannot be simply added, unless one of them is with S=0!
         assert(qt2.info.sym.is_zero());
         assert(qt3a.info.sym.is_zero());
         assert(iftrans == false);
   
         //auto sym2 = iftrans? -qt2.info.sym : qt2.info.sym;
         //qsym sym = qt3a.info.sym + sym2;
         qsym sym({3,0,0});

         auto qext = iftrans? qt2.info.qcol : qt2.info.qrow; 
         auto qint = iftrans? qt2.info.qrow : qt2.info.qcol;
         auto dext = qt2.dir_row(); // see the comment in stensor2<Tm>::H()
         auto dint = qt2.dir_col();
         stensor3su2<Tm> qt3;
         if(cpos == "l"){
            assert(qt3a.dir_row() == !dint);
            assert(qt3a.info.qrow == qint);
            direction3 dir = {dext, qt3a.dir_col(), qt3a.dir_mid()};
            qt3.init(sym, qext, qt3a.info.qcol, qt3a.info.qmid, dir, qt3a.info.couple);
            contract_qt3_qt2_su2_info_l(qt3a.info, qt3a.data(), qt2.info, qt2.data(),
                  qt3.info, qt3.data(), iftrans);
         }else if(cpos == "r"){
            assert(qt3a.dir_col() == !dint);
            assert(qt3a.info.qcol == qint);
            direction3 dir = {qt3a.dir_row(), dext, qt3a.dir_mid()};
            qt3.init(sym, qt3a.info.qrow, qext, qt3a.info.qmid, dir);
            contract_qt3_qt2_info_r(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
                  qt3.info, qt3.data(), iftrans); 
/*
         }else if(cpos == "c"){
            assert(qt3a.dir_mid() == !dint);
            assert(qt3a.info.qmid == qint);
            direction3 dir = {qt3a.dir_row(), qt3a.dir_col(), dext};
            qt3.init(sym, qt3a.info.qrow, qt3a.info.qcol, qext, dir);
            contract_qt3_qt2_info_c(qt3a.info, qt3a.data(), qt2.info, qt2.data(), 
                  qt3.info, qt3.data(), iftrans); 
*/
         }else{
            std::cout << "error: no such case in contract_qt3_qt2_su2! cpos=" 
               << cpos << std::endl;
            exit(1);
         }
         return qt3;
      }

   // formula: qt3(r,c,m) = \sum_x qt2(r,x)*qt3a(x,c,m) ; iftrans=false 
   // 		       = \sum_x qt2(x,r)*qt3a(x,c,m) ; iftrans=true
   //
   //  r/	m 
   //   *  |     = [m](r,c) = op(r,x) A[m](x,c) = <mr|o|c>
   //  x\--*--c
   template <typename Tm>
      void contract_qt3_qt2_su2_info_l(const qinfo3su2<Tm>& qt3a_info,
            const Tm* qt3a_data,	
            const qinfo2su2<Tm>& qt2_info,
            const Tm* qt2_data,
            const qinfo3su2<Tm>& qt3_info,
            Tm* qt3_data,
            const bool iftrans=false){
         const Tm alpha = 1.0, beta = 0.0;
         const char* transa = iftrans? "T" : "N";
         int br, bc, bm, tsi;
         for(int i=0; i<qt3_info._nnzaddr.size(); i++){
            auto key = qt3_info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            size_t off3 = qt3_info.get_offset(br,bc,bm,tsi);
            Tm* blk3 = qt3_data + off3-1;
            int rdim = qt3_info.qrow.get_dim(br);
            int cdim = qt3_info.qcol.get_dim(bc);
            int mdim = qt3_info.qmid.get_dim(bm);
            size_t size = rdim*cdim*mdim;
            bool ifzero = true;
            // find contracted index for
            // qt3(r,c,m) = \sum_x qt2(r,x)*qt3a(x,c,m) ; iftrans=false 
            // 	        = \sum_x qt2(x,r)*qt3a(x,c,m) ; iftrans=true
            const auto& nnzbx = iftrans? qt2_info._bc2br[br] : qt2_info._br2bc[br]; 
            for(const auto& bx : nnzbx){
               size_t off3a = qt3a_info.get_offset(bx,bc,bm,tsi);
               if(off3a == 0) continue;
               ifzero = false;
               size_t off2 = iftrans? qt2_info.get_offset(bx,br) : qt2_info.get_offset(br,bx);
               const Tm* blk3a = qt3a_data + off3a-1;
               const Tm* blk2 = qt2_data + off2-1;
               int xdim = qt3a_info.qrow.get_dim(bx);
               int LDA = iftrans? xdim : rdim;
               int cmdim = cdim*mdim;
               linalg::xgemm(transa, "N", rdim, cmdim, xdim, alpha,
                     blk2, LDA, blk3a, xdim, beta,
                     blk3, rdim); 
            } // bx 
            if(ifzero) memset(blk3, 0, size*sizeof(Tm));
         } // i
      }

   // formula: qt3(r,c,m) = \sum_x qt2(c,x)*qt3a(r,x,m) ; iftrans=false 
   // 		       = \sum_x qt2(x,c)*qt3a(r,x,m) ; iftrans=true
   //
   //     m  \ c/r
   //     |  *  = [m](r,c) = A[m](r,x) op(c,x) [permuted contraction (AO^T)]
   //  r--*--/ x/c
   template <typename Tm>
      void contract_qt3_qt2_info_r(const qinfo3su2<Tm>& qt3a_info,
            const Tm* qt3a_data,	
            const qinfo2su2<Tm>& qt2_info,
            const Tm* qt2_data,
            const qinfo3su2<Tm>& qt3_info,
            Tm* qt3_data,
            const bool iftrans=false){
         const Tm alpha = 1.0, beta = 0.0;
         const char* transb = iftrans? "N" : "T";
         int br, bc, bm, tsi;
         for(int i=0; i<qt3_info._nnzaddr.size(); i++){
            auto key = qt3_info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            size_t off3 = qt3_info.get_offset(br,bc,bm,tsi);
            Tm* blk3 = qt3_data + off3-1;
            int rdim = qt3_info.qrow.get_dim(br);
            int cdim = qt3_info.qcol.get_dim(bc);
            int mdim = qt3_info.qmid.get_dim(bm);
            size_t size = rdim*cdim*mdim;
            bool ifzero = true;
            // find contracted index for
            // qt3(r,c,m) = \sum_x qt2(c,x)*qt3a(r,x,m) ; iftrans=false 
            //     	     = \sum_x qt2(x,c)*qt3a(r,x,m) ; iftrans=true
            const auto& nnzbx = iftrans? qt2_info._bc2br[bc] : qt2_info._br2bc[bc];
            for(const auto& bx : nnzbx){
               size_t off3a = qt3a_info.get_offset(br,bx,bm,tsi);
               if(off3a == 0) continue;
               ifzero = false;
               size_t off2 = iftrans? qt2_info.get_offset(bx,bc) : qt2_info.get_offset(bc,bx); 
               const Tm* blk3a = qt3a_data + off3a-1;
               const Tm* blk2 = qt2_data + off2-1;
               int xdim = qt3a_info.qcol.get_dim(bx);
               int LDB = iftrans? xdim : cdim;
               int rcdim = rdim*cdim;
               int rxdim = rdim*xdim;
               for(int im=0; im<mdim; im++){
                  const Tm* blk3a_im = blk3a + im*rxdim;
                  Tm* blk3_im = blk3 + im*rcdim;
                  linalg::xgemm("N", transb, rdim, cdim, xdim, alpha,
                        blk3a_im, rdim, blk2, LDB, beta,
                        blk3_im, rdim);
               } // im
            } // bx
            if(ifzero) memset(blk3, 0, size*sizeof(Tm));
         } // i
      }

/*
   // formula: qt3(r,c,m) = \sum_x qt2(m,x)*qt3a(r,c,x) ; iftrans=false 
   // 		       = \sum_x qt2(x,m)*qt3a(r,c,x) ; iftrans=true
   //
   //     |m/r
   //     *	 
   //     |x/c  = [m](r,c) = B(m,x) A[x](r,c) [mostly used for op*wf]
   //  r--*--c
   template <typename Tm>
      void contract_qt3_qt2_info_c(const qinfo3su2<Tm>& qt3a_info,
            const Tm* qt3a_data,	
            const qinfo2su2<Tm>& qt2_info,
            const Tm* qt2_data,
            const qinfo3su2<Tm>& qt3_info,
            Tm* qt3_data,
            const bool iftrans=false){
         const Tm alpha = 1.0, beta = 0.0;
         const char* transb = iftrans? "N" : "T";
         int br, bc, bm;
         for(int i=0; i<qt3_info._nnzaddr.size(); i++){
            int idx = qt3_info._nnzaddr[i];
            qt3_info._addr_unpack(idx,br,bc,bm);
            size_t off3 = qt3_info._offset[idx];
            Tm* blk3 = qt3_data + off3-1;
            int rdim = qt3_info.qrow.get_dim(br);
            int cdim = qt3_info.qcol.get_dim(bc);
            int mdim = qt3_info.qmid.get_dim(bm);
            size_t size = rdim*cdim*mdim;
            bool ifzero = true;
            // find contracted index for
            // qt3(r,c,m) = \sum_x qt2(m,x)*qt3a(r,c,x) ; iftrans=false 
            // 	    = \sum_x qt2(x,m)*qt3a(r,c,x) ; iftrans=true
            int bx = iftrans? qt2_info._bc2br[bm] : qt2_info._br2bc[bm];
            if(bx != -1){
               size_t off3a = qt3a_info._offset[qt3a_info._addr(br,bc,bx)];
               if(off3a != 0){
                  ifzero = false;
                  int jdx = iftrans? qt2_info._addr(bx,bm) : qt2_info._addr(bm,bx);
                  size_t off2 = qt2_info._offset[jdx];
                  const Tm* blk3a = qt3a_data + off3a-1;
                  const Tm* blk2 = qt2_data + off2-1;
                  int xdim = qt3a_info.qmid.get_dim(bx);
                  int LDB = iftrans? xdim : mdim;
                  int rcdim = rdim*cdim;
                  linalg::xgemm("N", transb, rcdim, mdim, xdim, alpha,
                        blk3a, rcdim, blk2, LDB, beta,
                        blk3, rcdim);
               }
            }
            if(ifzero) memset(blk3, 0, size*sizeof(Tm));
         } // i
      }
*/
} // ctns

#endif
