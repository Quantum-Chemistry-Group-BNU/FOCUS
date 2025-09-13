#ifndef SWEEP_TWODOT_GUESS_H
#define SWEEP_TWODOT_GUESS_H

namespace ctns{

   // initial guess for next site within the bond
   template <typename Qm, typename Tm>
      void twodot_guess_psi(const std::string superblock,
            comb<Qm,Tm>& icomb,
            const directed_bond& dbond,
            const linalg::matrix<Tm>& vsol,
            qtensor4<Qm::ifabelian,Tm>& wf,
            const qtensor2<Qm::ifabelian,Tm>& rot){
         const bool debug = false;
         if(debug) std::cout << "ctns::twodot_guess_psi superblock=" << superblock << std::endl;
         int nroots = vsol.cols();
         icomb.cpsi.clear();
         icomb.cpsi.resize(nroots);
         if(superblock == "lc1"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               //------------------------------------------
               // Two-dot case: simply use cwf[alpha,r,c2]
               //------------------------------------------
               // wf4[l,r,c1,c2] => wf2[lc1,c2r]
               auto wf2 = wf.merge_lc1_c2r();
               // rot.H()[alpha,lc1]*wf2[lc1,c2r] => cwf[alpha,c2r]
               auto cwf = rot.H().dot(wf2); 
               // cwf[alpha,c2r] => psi[alpha,r,c2]
               auto psi = cwf.split_cr(wf.info.qver, wf.info.qcol);
               //------------------------------------------
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "c2r"){

            for(int i=0; i<vsol.cols(); i++){
               wf.from_array(vsol.col(i));
               //------------------------------------------
               // Two-dot case: simply use cwf[l,alpha,c1]
               //------------------------------------------
               // wf4[l,r,c1,c2] => wf2[lc1,c2r]
               auto wf2 = wf.merge_lc1_c2r();
               // wf2[lc1,c2r]*rot.H()[c2r,alpha] => cwf[lc1,alpha]
               auto cwf = wf2.dot(rot.H());
               // cwf[lc1,alpha] => cwf[l,alpha,c1]
               auto psi = cwf.split_lc(wf.info.qrow, wf.info.qmid);
               //------------------------------------------
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "lr"){

            assert(Qm::ifabelian);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               //-------------------------------------------
               // Two-dot case: simply use cwf[alpha,c2,c1]
               //-------------------------------------------
               // wf4[l,r,c1,c2] => wf2[lr,c1c2]
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2();
               // rot.H()[alpha,lr]*wf3[lr,c1c2] => cwf[alpha,c1c2]
               auto cwf = rot.H().dot(wf2);
               // cwf[alpha,c1c2] => cwf[alpha,c2,c1] 
               auto psi = cwf.split_cr(wf.info.qmid, wf.info.qver);
               //-------------------------------------------
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "c1c2"){

            assert(Qm::ifabelian);
            for(int i=0; i<vsol.cols(); i++){
               wf.from_array(vsol.col(i));
               //----------------------------------------------
               // Two-dot case: simply use cwf[l,r,alpha]
               //----------------------------------------------
               wf.permCR_signed();
               // wf4[l,c1,c2,r] => wf2[lr,c1c2]
               auto wf2 = wf.merge_lr_c1c2();
               // wf2[lr,c1c2]*rot.H()[c1c2,alpha] => cwf[lr,alpha]
               auto cwf = wf2.dot(rot.H());
               // cwf[lr,alpha] => psi[l,r,alpha]
               auto psi = cwf.split_lr(wf.info.qrow, wf.info.qcol);
               // revert ordering of the underlying basis
               psi.permCR_signed(); 
               //----------------------------------------------
               icomb.cpsi[i] = std::move(psi); // psi on backbone
            }

         } // superblock
      }

   template <typename Qm, typename Tm>
      void twodot_guess_v0(const comb<Qm,Tm>& icomb, 
            const directed_bond& dbond,
            const size_t ndim,
            const int neig,
            qtensor4<Qm::ifabelian,Tm>& wf,
            std::vector<Tm>& v0,
            const bool debug=true){
         if(debug) std::cout << "ctns::twodot_guess_v0 ";
         auto pdx0 = icomb.topo.rindex.at(dbond.p0);
         auto pdx1 = icomb.topo.rindex.at(dbond.p1);
         assert(icomb.cpsi.size() == neig);
         v0.resize(ndim*neig);
         if(dbond.forward){
            if(!dbond.is_cturn()){

               if(debug) std::cout << "|lc1>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[l,a,c1] => cwf[lc1,a]
                  auto cwf = icomb.cpsi[i].recouple_lc().merge_lc(); 
                  // cwf[lc1,a]*r[a,r,c2] => wf3[lc1,r,c2]
                  auto wf3 = contract_qt3_qt2("l",icomb.sites[pdx1],cwf); 
                  // wf3[lc1,r,c2] => wf4[l,r,c1,c2]
                  auto wf4 = wf3.split_lc1(wf.info.qrow, wf.info.qmid);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               assert(Qm::ifabelian);
               //
               //     c2
               //      |
               // c1---p1 
               //      |
               //  l---p0---r
               //     [psi]
               //
               if(debug) std::cout << "|lr>(comb)" << std::endl;
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

               if(debug) std::cout << "|c2r>" << std::endl;
               for(int i=0; i<neig; i++){
                  // psi[a,r,c2] => cwf[a,c2r]
                  auto cwf = icomb.cpsi[i].recouple_cr().merge_cr();
                  // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
                  auto wf3 = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.P());
                  // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
                  auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
                  assert(wf4.size() == ndim);
                  wf4.to_array(&v0[ndim*i]);
               }

            }else{

               assert(Qm::ifabelian);
               //
               //     c2
               //      |
               // c1---p0 [psi]
               //      |
               //  l---p1---r
               //
               if(debug) std::cout << "|c1c2>(comb)" << std::endl;
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
