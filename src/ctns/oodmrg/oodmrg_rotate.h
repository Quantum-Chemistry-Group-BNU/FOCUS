#ifndef OODMRG_ROTATE_H
#define OODMRG_ROTATE_H

#include "../../core/spin.h"

namespace ctns{

   const bool debug_twodot_rotate = true;
   extern const bool debug_twodot_rotate;

   // su2 case
   template <bool ifab, typename Tm>
      void twodot_rotate(const std::vector<Tm>& v0, 
            std::vector<Tm>& vr, 
            qtensor4<ifab,Tm>& wf, 
            const double theta){
         if(debug_twodot_rotate){
            std::cout << "\nctns::twodot_rotate(su2): theta=" << theta << std::endl;
            wf.print("wf",1);
         }
         assert(!ifab);
         assert(v0.size() == vr.size());
         // this function only works for singlet embedding case
         auto sym = wf.info.sym;
         if(sym.ts() != 0){
            std::cout << "error in twodot_rotate(su2): only support singlet, but ts=" << sym.ts() << std::endl;
            exit(1);
         }  
         double c = std::cos(theta);
         double s = std::sin(theta);
         double c2 = c*c, s2 = s*s, cs = c*s;
         // clear
         memset(vr.data(), 0, vr.size()*sizeof(Tm));
         int br, bc, bm, bv, tsi, tsj;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            auto key = wf.info._nnzaddr[i];
            br = std::get<0>(key);
            bc = std::get<1>(key);
            bm = std::get<2>(key);
            bv = std::get<3>(key);
            tsi = std::get<4>(key);
            tsj = std::get<5>(key);
            int tsl  = wf.info.qrow.get_sym(br).ts();
            int tsr  = wf.info.qcol.get_sym(bc).ts();
            int tsc1 = wf.info.qmid.get_sym(bm).ts();
            int tsc2 = wf.info.qmid.get_sym(bv).ts();
            auto blk4 = wf(br,bc,bm,bv,tsi,tsj);
            assert(tsi == tsj); // for singlet wavefunction
            size_t size = blk4.size();
            size_t offset = wf.info.get_offset(br,bc,bm,bv,tsi,tsj);
            assert(offset > 0);
            offset -= 1;
            // let {0,1,2} refers to {|N,S>}={|0,0>,|2,0>,|1,1/2>} as defined in init_phys.h
            // in total 9 subcases:
            // case-1: {(0,0)},{(1,1)} - no transformation is needed
            if((bm == 0 and bv == 0) or
               (bm == 1 and bv == 1)){
               std::cout << "case-1" << std::endl;
               linalg::xcopy(size, &v0[offset], &vr[offset]); 
            }
            // case-2: {(2,0),(0,2)},{(2,1),(1,2)}
            if((bm == 2 and bv == 0) or
               (bm == 2 and bv == 1)){
               std::cout << "case-2a" << std::endl;
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm,tsl,tsl);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size, c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, s, &v0[offset1], &vr[offset]); 
            }
            if((bm == 0 and bv == 2) or
               (bm == 1 and bv == 2)){
               std::cout << "case-2b" << std::endl;
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm,tsr,tsr);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size,  c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, -s, &v0[offset1], &vr[offset]); 
            }
            // case-3: {(1,0),(0,1),(2,2)}
            if((bm == 1 and bv == 0)){
               std::cout << "case-3a" << std::endl;
               assert(tsl==tsr && tsl==tsi && tsl==tsj);
               size_t offset0 = wf.info.get_offset(br,bc,1,0,tsi,tsj);
               size_t offset1 = wf.info.get_offset(br,bc,0,1,tsi,tsj);
               assert(offset0 > 0 and offset1 > 0);
               offset0 -= 1;
               offset1 -= 1;
               linalg::xaxpy(size, c2, &v0[offset0], &vr[offset]);
               linalg::xaxpy(size, s2, &v0[offset1], &vr[offset]);
               for(int tshp=std::abs(tsl-1); tshp<=tsl+1; tshp+=2){
                  size_t offset2 = wf.info.get_offset(br,bc,2,2,tshp,tshp);
                  Tm fac = std::sqrt(2)*cs*std::sqrt((tshp+1.0)/(tsl+1.0)/2.0);
                  if((3*tsl+3+tshp)%2 == 1) fac = -fac; // additional sign
                  linalg::xaxpy(size, fac, &v0[offset2], &vr[offset]);
               }
            }
            if((bm == 0 and bv == 1)){
               std::cout << "case-3b" << std::endl;
               assert(tsl==tsr && tsl==tsi && tsl==tsj);
               size_t offset0 = wf.info.get_offset(br,bc,1,0,tsi,tsj);
               size_t offset1 = wf.info.get_offset(br,bc,0,1,tsi,tsj);
               assert(offset0 > 0 and offset1 > 0);
               offset0 -= 1;
               offset1 -= 1;
               linalg::xaxpy(size, s2, &v0[offset0], &vr[offset]);
               linalg::xaxpy(size, c2, &v0[offset1], &vr[offset]);
               for(int tshp=std::abs(tsl-1); tshp<=tsl+1; tshp+=2){
                  size_t offset2 = wf.info.get_offset(br,bc,2,2,tshp,tshp);
                  Tm fac = -std::sqrt(2)*cs*std::sqrt((tshp+1.0)/(tsl+1.0)/2.0);
                  if((3*tsl+3+tshp)%2 == 1) fac = -fac; // additional sign
                  linalg::xaxpy(size, fac, &v0[offset2], &vr[offset]);
               }
            }
            if((bm == 2 and bv == 2)){
               std::cout << "case-3c: tsl,tsr,tsc1,tsc2,tsi,tsj=" 
                  << tsl << "," << tsr << "," << tsc1 << "," << tsc2 << ","
                  << tsi << "," << tsj << std::endl;
               if(tsl == tsr){
                  size_t offset0 = wf.info.get_offset(br,bc,1,0,tsl,tsl);
                  size_t offset1 = wf.info.get_offset(br,bc,0,1,tsl,tsl);
                  assert(offset0 > 0 and offset1 > 0);
                  offset0 -= 1;
                  offset1 -= 1;
                  Tm fac = -std::sqrt(2)*cs*std::sqrt((tsi+1.0)/(tsl+1.0)/2.0);
                  if((3*tsl+3+tsi)%2 == 1) fac = -fac; // additional sign
                  linalg::xaxpy(size,  fac, &v0[offset0], &vr[offset]);
                  linalg::xaxpy(size, -fac, &v0[offset1], &vr[offset]);
               }
               int tsmin = std::min(tsl,tsr);
               for(int tshp=std::abs(tsmin-1); tshp<=tsmin+1; tshp+=2){
                  if(!fock::spin_triangle(tsl,1,tshp) or !fock::spin_triangle(tsr,1,tshp)) continue;   
                  std::cout << "tshp=" << tshp << std::endl;
                  // S[c1c2]=0
                  Tm fac = 0.0;
                  if(tsl == tsr){
                     fac += std::cos(2*theta)*fock::racah(tsl,1,tsr,1,tsi,0)*fock::racah(tsl,1,tsr,1,tshp,0);
                  }
                  // check Triangle(tshp,tsr,1)
                  std::cout << "lzd" << std::endl;
                  std::cout << "lzd fac=" << fac << std::endl;
                  // S[c1c2]=1
                  fac += fock::racah(tsl,1,tsr,1,tsi,2)*fock::racah(tsl,1,tsr,1,tshp,2);
                  std::cout << "lzd fac=" << fac << std::endl;
                  fac *= std::sqrt((tsi+1.0)*(tshp+1.0));
                  // multiply wavefunction
                  size_t offset2 = wf.info.get_offset(br,bc,2,2,tshp,tshp);
                  assert(offset2 > 0);
                  offset2 -= 1;
                  linalg::xaxpy(size, fac, &v0[offset2], &vr[offset]);
               } // tshp
            }
         } // i
         // debug by checking the norm of the rotated wavefunction,
         // which should be identitcal to the unrotated one.
         if(debug_twodot_rotate){
            double norm0 = linalg::xnrm2(v0.size(), v0.data());
            double norm1 = linalg::xnrm2(vr.size(), vr.data());
            if(std::abs(norm0-norm1)>1.e-10){
               std::cout << "error in twodot_rotate(su2):"
                  << " norm0=" << norm0 
                  << " norm1=" << norm1
                  << " diff=" << norm0-norm1
                  << std::endl;
               exit(1);
            }
         }
      }

   // Abelian
   template <bool ifab, typename Tm, std::enable_if_t<ifab,int> = 0>
      void twodot_rotate(const std::vector<Tm>& v0, 
            std::vector<Tm>& vr, 
            stensor4<Tm>& wf, 
            const double theta){
         if(debug_twodot_rotate){
            std::cout << "\nctns::twodot_rotate: theta=" << theta << std::endl;
            wf.print("wf",1);
         }
         assert(v0.size() == vr.size());
         double c = std::cos(theta);
         double s = std::sin(theta);
         double c2 = c*c, s2 = s*s, cs = c*s;
         // encode the unitary matrix for the subspace by a dictionary
         std::map<std::tuple<int,int,int>,double> udict = {
            {std::make_tuple(1,0,0), c2},{std::make_tuple(0,1,0), s2},{std::make_tuple(2,3,0),-cs},{std::make_tuple(3,2,0), cs},
            {std::make_tuple(1,0,1), s2},{std::make_tuple(0,1,1), c2},{std::make_tuple(2,3,1), cs},{std::make_tuple(3,2,1),-cs},
            {std::make_tuple(1,0,2), cs},{std::make_tuple(0,1,2),-cs},{std::make_tuple(2,3,2), c2},{std::make_tuple(3,2,2), s2},
            {std::make_tuple(1,0,3),-cs},{std::make_tuple(0,1,3), cs},{std::make_tuple(2,3,3), s2},{std::make_tuple(3,2,3), c2}
         };
         // clear
         memset(vr.data(), 0, vr.size()*sizeof(Tm));
         int br, bc, bm, bv;
         for(int i=0; i<wf.info._nnzaddr.size(); i++){
            int idx = wf.info._nnzaddr[i];
            wf.info._addr_unpack(idx,br,bc,bm,bv);
            auto blk4 = wf(br,bc,bm,bv);
            size_t size = blk4.size();
            size_t offset = wf.info.get_offset(br,bc,bm,bv);
            assert(offset > 0);
            offset -= 1;
            // let {0,1,2,3} refers to {|0>,|2>,|a>,|b>} as defined in init_phys.h
            // in total 16 subcases:
            // case-1: {(0,0)},{(1,1)},{(2,2)},{(3,3)} - no transformation is needed
            if((bm == 0 and bv == 0) or
               (bm == 1 and bv == 1) or
               (bm == 2 and bv == 2) or
               (bm == 3 and bv == 3)){
               linalg::xcopy(size, &v0[offset], &vr[offset]); 
            }
            // case-2: {(2,0),(0,2)},{(3,0),(0,3)},{(2,1),(1,2)},{(3,1),(1,3)}
            if((bm == 2 and bv == 0) or
               (bm == 3 and bv == 0) or
               (bm == 2 and bv == 1) or
               (bm == 3 and bv == 1)){
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size, c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, s, &v0[offset1], &vr[offset]); 
            }
            if((bm == 0 and bv == 2) or
               (bm == 0 and bv == 3) or 
               (bm == 1 and bv == 2) or
               (bm == 1 and bv == 3)){
               size_t offset1 = wf.info.get_offset(br,bc,bv,bm);
               assert(offset1 > 0);
               offset1 -= 1;
               linalg::xaxpy(size,  c, &v0[offset] , &vr[offset]);
               linalg::xaxpy(size, -s, &v0[offset1], &vr[offset]); 
            }
            // case-3: {(1,0),(0,1),(2,3),(3,2)}
            if((bm == 1 and bv == 0) or
               (bm == 0 and bv == 1) or
               (bm == 2 and bv == 3) or
               (bm == 3 and bv == 2)){
               size_t offset0 = wf.info.get_offset(br,bc,1,0);
               size_t offset1 = wf.info.get_offset(br,bc,0,1);
               size_t offset2 = wf.info.get_offset(br,bc,2,3);
               size_t offset3 = wf.info.get_offset(br,bc,3,2);
               assert(offset0 > 0 and offset1 > 0 and offset2 > 0 and offset3 > 0);
               offset0 -= 1;
               offset1 -= 1;
               offset2 -= 1;
               offset3 -= 1;
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,0)), &v0[offset0], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,1)), &v0[offset1], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,2)), &v0[offset2], &vr[offset]);
               linalg::xaxpy(size, udict.at(std::make_tuple(bm,bv,3)), &v0[offset3], &vr[offset]);
            }
         } // i
         // debug by checking the norm of the rotated wavefunction,
         // which should be identitcal to the unrotated one.
         if(debug_twodot_rotate){
            double norm0 = linalg::xnrm2(v0.size(), v0.data());
            double norm1 = linalg::xnrm2(vr.size(), vr.data());
            if(std::abs(norm0-norm1)>1.e-10){
               std::cout << "error in twodot_rotate:"
                  << " norm0=" << norm0 
                  << " norm1=" << norm1
                  << " diff=" << norm0-norm1
                  << std::endl;
               exit(1);
            }
         }
      }

} // ctns

#endif
