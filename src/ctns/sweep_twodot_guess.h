#ifndef SWEEP_TWODOT_GUESS_H
#define SWEEP_TWODOT_GUESS_H

#include "oper_functors.h"

namespace ctns{

   const bool debug_twodot_guess = true;
   extern const bool debug_twodot_guess;

   template <typename Km>
      void twodot_guess(comb<Km>& icomb, 
            const directed_bond& dbond,
            const size_t ndim,
            const int neig,
            stensor4<typename Km::dtype>& wf,
            std::vector<typename Km::dtype>& v0){
         if(debug_twodot_guess) std::cout << "ctns::twodot_guess ";
         auto pdx0 = icomb.topo.rindex.at(dbond.p0);
         auto pdx1 = icomb.topo.rindex.at(dbond.p1);
         assert(icomb.cpsi.size() == neig);
         v0.resize(ndim*neig);
         if(dbond.forward){
            if(!dbond.is_cturn()){

               if(debug_twodot_guess) std::cout << "|lc1>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[l,a,c1] => cwf[lc1,a]
                  auto cwf = icomb.cpsi[i].merge_lc(); 
                  // cwf[lc1,a]*r[a,r,c2] => wf3[lc1,r,c2]
                  auto wf3 = contract_qt3_qt2("l",icomb.sites[pdx1],cwf); 
                  // wf3[lc1,r,c2] => wf4[l,r,c1,c2]
                  auto wf4 = wf3.split_lc1(wf.info.qrow, wf.info.qmid);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               //
               //     c2
               //      |
               // c1---p1 
               //      |
               //  l---p0---r
               //     [psi]
               //
               if(debug_twodot_guess) std::cout << "|lr>(comb)" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[l,r,a] => cwf[lr,a]		 
                  auto cwf = icomb.cpsi[i].merge_lr(); // on backone
                  // r[a,c2,c1] => r[a,c1c2], cwf[lr,a]*r[a,c1c2] => wf2[lr,c1c2]
                  auto wf2 = cwf.dot(icomb.sites[pdx1].merge_cr());
                  // wf2[lr,c1c2] => wf4[l,r,c1,c2] 
                  auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            } // cturn
         }else{
            if(!dbond.is_cturn()){

               if(debug_twodot_guess) std::cout << "|c2r>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[a,r,c2] => cwf[a,c2r]
                  auto cwf = icomb.cpsi[i].merge_cr();
                  // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
                  auto wf3 = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.T());
                  // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
                  auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               //
               //     c2
               //      |
               // c1---p0 [psi]
               //      |
               //  l---p1---r
               //
               if(debug_twodot_guess) std::cout << "|c1c2>(comb)" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[a,c2,c1] => cwf[a,c1c2]
                  auto cwf = icomb.cpsi[i].merge_cr(); // on branch
                  // l[l,r,a] => l[lr,a], l[lr,a]*cwf[a,c1c2] => wf2[lr,c1c2]
                  auto wf2 = icomb.sites[pdx0].merge_lr().dot(cwf);
                  // wf2[lr,c1c2] => wf4[l,r,c1,c2]
                  auto wf4 = wf2.split_lr_c1c2(wf.info.qrow, wf.info.qcol, wf.info.qmid, wf.info.qver);
                  wf4.permCR_signed(); // back to backbone
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            } // cturn
         } // forward
      }

} // ctns

#endif
