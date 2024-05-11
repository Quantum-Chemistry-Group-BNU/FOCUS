#ifndef SWEEP_ONEDOT_GUESS_H
#define SWEEP_ONEDOT_GUESS_H

namespace ctns{

   // initial guess for next site within the bond
   template <typename Qm, typename Tm>
      void onedot_guess_psi(const std::string superblock,
            comb<Qm,Tm>& icomb,
            const directed_bond& dbond,
            const linalg::matrix<Tm>& vsol,
            qtensor3<Qm::ifabelian,Tm>& wf,
            const qtensor2<Qm::ifabelian,Tm>& rot){
         const bool debug = false;
         if(debug) std::cout << "ctns::onedot_guess_psi superblock=" << superblock << std::endl;
         const auto& pdx0 = icomb.topo.rindex.at(dbond.p0);
         const auto& pdx1 = icomb.topo.rindex.at(dbond.p1);
         int nroots = vsol.cols();
         icomb.cpsi.clear();
         icomb.cpsi.resize(nroots);
         if(superblock == "lc"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto cwf = rot.H().dot(wf.recouple_lc().merge_lc()); // <-W[alpha,r]->
               auto psi = contract_qt3_qt2("l",icomb.sites[pdx1],cwf);
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "lr"){

            assert(Qm::ifabelian);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               wf.permCR_signed();
               auto cwf = rot.H().dot(wf.merge_lr()); // <-W[alpha,r]->
               auto psi = contract_qt3_qt2("l",icomb.sites[pdx1],cwf);
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "cr"){

            auto cturn = dbond.is_cturn(); 
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto cwf = wf.recouple_cr().merge_cr().dot(rot.H()); // <-W[l,alpha]->
               if(!cturn){
                  auto psi = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.P());
                  icomb.cpsi[i] = std::move(psi);
               }else{
                  // special treatment of the propagation downside to backbone
                  auto psi = contract_qt3_qt2("c",icomb.sites[pdx0],cwf.P());
                  psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
                  icomb.cpsi[i] = std::move(psi);
               }
            }

         } // superblock
      }

} // ctns

#endif
