#ifndef STENSOR3SU2_H
#define STENSOR3SU2_H

#include "qtensor3.h"

namespace ctns{

   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      void qtensor3<ifab,Tm>::print(const std::string name, const int level) const{
         std::cout << "qtensor3: " << name << " own=" << own << " _data=" << _data << std::endl; 
         info.print(name);
         int br, bc, bm, tsi;
         for(int i=0; i<info._nnzaddr.size(); i++){
            auto key = info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            tsi = std::get<3>(key);
            const auto blk = (*this)(br,bc,bm,tsi);
            if(level >= 1){
               std::cout << "i=" << i << " block["
                  << br << ":" << info.qrow.get_sym(br) << "," 
                  << bc << ":" << info.qcol.get_sym(bc) << "," 
                  << bm << ":" << info.qmid.get_sym(bm) << ";"
                  << tsi << "]" 
                  << " dim0,dim1,dim2=(" 
                  << blk.dim0 << "," 
                  << blk.dim1 << ","
                  << blk.dim2 << ")" 
                  << " size=" << blk._size 
                  << std::endl; 
               if(level >= 2) blk.print("blk_"+std::to_string(i));
            } // level>=1
         } // i
      }

   // get block by sym used in rcanon_CIcoeff
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      dtensor3<Tm> qtensor3<ifab,Tm>::get_rcf_symblk(const qsym qr, const qsym qc, const qsym qm) const{
         int br = info.qrow.existQ(qr);
         int bc = info.qcol.existQ(qc);
         int bm = info.qmid.existQ(qm);
         if(br == -1 || bc == -1 || bm == -1){
            return dtensor3<Tm>();
         }
         // CRcouple
         assert(info.couple == CRcouple);
         // For MPS site (sym=0), tsi is uniquely determined by ql
         int tsi = info.qrow.get_sym(br).ts(); 
         return (*this)(br,bc,bm,tsi);
      }

   /*
   // fix middle index (bm,im) - bm-th block, im-idx - composite index!
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor2<ifab,Tm> qtensor3<ifab,Tm>::fix_mid(const std::pair<int,int> mdx) const{
         // We only define this function for MPS site
         assert(info.sym.is_zero());
         int bm = mdx.first, im = mdx.second;
         assert(im < info.qmid.get_dim(bm));
         auto sym = info.qmid.get_sym(bm);
         auto symIn = std::get<2>(info.dir) ? qsym(3,-sym.ne(),sym.ts()) : qsym(3,sym.ne(),sym.ts());
         qtensor2<ifab,Tm> qt2(symIn, info.qrow, info.qcol,
               {std::get<0>(info.dir), std::get<1>(info.dir)});
         
         std::cout << "bm,im=" << bm << "," << im << std::endl;
         std::cout << "symIn=" << symIn << std::endl;
         qt2.print("qt2",2);
         std::cout << "lzd" << std::endl;

         int br, bc;
         for(int i=0; i<info._nnzaddr.size(); i++){
            auto key = qt2.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
         std::cout << "lzd1" << std::endl;
            auto blk2 = qt2(br,bc);
         std::cout << "lzd1x" << std::endl;
            if(info.couple == LCcouple){
               // coupling (lc)r: s[lc]=s[r] because sym is zero.
               int tsr = info.qcol.get_sym(bc).ts();
               const auto blk3 = (*this)(br,bc,bm,tsr);
               if(blk3.empty()) continue;
               linalg::xcopy(blk2.size(), blk3.get(im).data(), blk2.data());
            }else{
               // coupling l(cr)
               int tsl = info.qrow.get_sym(br).ts();
               const auto blk3 = (*this)(br,bc,bm,tsl);
               if(blk3.empty()) continue;
               linalg::xcopy(blk2.size(), blk3.get(im).data(), blk2.data());
            }
         } // br
         std::cout << "lzd2" << std::endl;
         return qt2;
      }
      */

   // recouple_lc qt3[l,r,c;lc] from qt3[l,r,c,cr]
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor3<ifab,Tm> qtensor3<ifab,Tm>::recouple_lc() const{
         if(info.couple == LCcouple) return *this;
         qtensor3<ifab,Tm> qt3(info.sym, info.qrow, info.qcol, info.qmid, info.dir, LCcouple);
         int tstot = info.sym.ts();
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            int br = std::get<0>(key); // l
            int bc = std::get<1>(key); // r
            int bm = std::get<2>(key); // c
            int tslc = std::get<3>(key);
            auto blk = qt3(br,bc,bm,tslc);
            int tsl = info.qrow.get_sym(br).ts();
            int tsr = info.qcol.get_sym(bc).ts();
            int tsc = info.qmid.get_sym(bm).ts();
            for(int tscr=std::abs(tsc-tsr); tscr<=tsc+tsr; tscr+=2){
               const auto blk2 = (*this)(br,bc,bm,tscr);
               if(blk2.empty()) continue;
               double fac = std::sqrt((tslc+1.0)*(tscr+1.0))*fock::wigner6j(tsl,tsc,tslc,tsr,tstot,tscr);
               int ts = tsl+tsc+tsr+tstot;
               if((ts/2)%2==1) fac *= -1;
               linalg::xaxpy(blk.size(), fac, blk2.data(), blk.data());
            } // tscr
         } // i
         return qt3;
      }
    
   // recouple_lc qt3[l,r,c;cr]
   template <bool ifab, typename Tm>
      template <bool y, std::enable_if_t<!y,int>>
      qtensor3<ifab,Tm> qtensor3<ifab,Tm>::recouple_cr() const{
         if(info.couple == CRcouple) return *this;
         qtensor3<ifab,Tm> qt3(info.sym, info.qrow, info.qcol, info.qmid, info.dir, CRcouple);
         int tstot = info.sym.ts();
         for(int i=0; i<qt3.info._nnzaddr.size(); i++){
            auto key = qt3.info._nnzaddr[i];
            int br = std::get<0>(key); // l
            int bc = std::get<1>(key); // r
            int bm = std::get<2>(key); // c
            int tscr = std::get<3>(key);
            auto blk = qt3(br,bc,bm,tscr);
            int tsl = info.qrow.get_sym(br).ts();
            int tsr = info.qcol.get_sym(bc).ts();
            int tsc = info.qmid.get_sym(bm).ts();
            for(int tslc=std::abs(tsl-tsc); tslc<=tsl+tsc; tslc+=2){
               const auto blk2 = (*this)(br,bc,bm,tslc);
               if(blk2.empty()) continue;
               double fac = std::sqrt((tslc+1.0)*(tscr+1.0))*fock::wigner6j(tsl,tsc,tslc,tsr,tstot,tscr);
               int ts = tsl+tsc+tsr+tstot;
               if((ts/2)%2==1) fac *= -1;
               linalg::xaxpy(blk.size(), fac, blk2.data(), blk.data());
            } // tslc
         } // i
         return qt3;
      }
 
} // ctns

#endif
