#ifndef SWEEP_TWODOT_GUESS_SU2_H
#define SWEEP_TWODOT_GUESS_SU2_H

namespace ctns{

   // initial guess for next site within the bond
   template <typename Qm, typename Tm>
      void twodot_guess_psi(const std::string superblock,
            comb<Qm,Tm>& icomb,
            const directed_bond& dbond,
            const linalg::matrix<Tm>& vsol,
            stensor4su2<Tm>& wf,
            const stensor2su2<Tm>& rot){
         const bool debug = false;
         if(debug) std::cout << "ctns::twodot_guess_psi(su2) superblock=" << superblock << std::endl;
         int nroots = vsol.cols();
         icomb.cpsi.clear();
         icomb.cpsi.resize(nroots);
         std::cout << "twodot_guess_psi not implemented yet!" << std::endl;
         exit(1);
/*
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

         } // superblock
*/
      }

   template <typename Qm, typename Tm>
      void twodot_guess_v0(comb<Qm,Tm>& icomb, 
            const directed_bond& dbond,
            const size_t ndim,
            const int neig,
            stensor4su2<Tm>& wf,
            std::vector<Tm>& v0){
         std::cout << "twodot_guess_v0 not implemented for su2 yet!" << std::endl;
         exit(1);
/*
         const bool debug = true;
         if(debug) std::cout << "ctns::twodot_guess(su2) ";
         auto pdx0 = icomb.topo.rindex.at(dbond.p0);
         auto pdx1 = icomb.topo.rindex.at(dbond.p1);
         assert(icomb.cpsi.size() == neig);
         v0.resize(ndim*neig);
         if(dbond.forward){

            if(debug) std::cout << "|lc1>" << std::endl;
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

            if(debug) std::cout << "|c2r>" << std::endl;
            for(int i=0; i<neig; i++){
               // psi[a,r,c2] => cwf[a,c2r]
               auto cwf = icomb.cpsi[i].merge_cr();
               // l[l,a,c1]*cwf[a,c2r] => wf3[l,c2r,c1]
               auto wf3 = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.P());
               // wf3[l,c2r,c1] => wf4[l,r,c1,c2] 
               auto wf4 = wf3.split_c2r(wf.info.qver, wf.info.qcol);
               assert(wf4.size() == ndim);
               wf4.to_array(&v0[ndim*i]);
            }

         } // forward
*/      
      }

} // ctns

#endif
