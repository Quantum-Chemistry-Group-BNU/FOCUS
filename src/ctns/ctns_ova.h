#ifndef CTNS_OVA_H
#define CTNS_OVA_H

/*
   Algorithms for CTNS:

   0. rcanon_check: check sites in right canonical form (RCF)

   1. get_Smat: <CTNS[i]|CTNS[j]> 
*/

#include "../core/onspace.h"
#include "../core/analysis.h"
#include "ctns_comb.h"

namespace ctns{

   // Algorithm 0:
   // Check right canonical form
   template <typename Qm, typename Tm>
      void rcanon_check(const comb<Qm,Tm>& icomb,
            const double thresh_ortho,
            const bool ifortho=true){
         auto t0 = tools::get_time();
         std::cout << "\nctns::rcanon_check thresh_ortho=" 
            << std::scientific << std::setprecision(3) << thresh_ortho 
            << std::endl;
         // loop over all sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto p = icomb.topo.rcoord[idx];
            // check right canonical form -> A*[l'cr]A[lcr] = w[l'l] = Id
            auto qt2 = contract_qt3_qt3("cr", icomb.sites[idx], icomb.sites[idx]);
            double maxdiff = qt2.check_identityMatrix(thresh_ortho, false);
            int Dtot = qt2.info.qrow.get_dimAll();
            std::cout << " idx=" << idx << " node=" << p << " Drow=" << Dtot 
               << " maxdiff=" << std::scientific << maxdiff << std::endl;
            if(ifortho && (maxdiff>thresh_ortho)){
               tools::exit("error: deviate from identity matrix!");
            }
         } // idx
         // rwfuns
         auto wf2 = icomb.get_wf2();
         wf2.print("wf2",2);
         auto qt2 = contract_qt2_qt2(wf2, wf2.H());
         double maxdiff = qt2.check_identityMatrix(thresh_ortho, false);
         int Dtot = qt2.info.qrow.get_dimAll();
         std::cout << " rwfuns: nroots=" << Dtot 
            << " maxdiff=" << std::scientific << maxdiff << std::endl;
         if(ifortho && (maxdiff>thresh_ortho)){
            tools::exit("error: deviate from identity matrix!");
         }
         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_check", t0, t1);
      }

   // Algorithm 1:
   // <CTNS[i]|CTNS[j]>: compute by a typical loop for right canonical form 
   template <typename Qm, typename Tm>
      linalg::matrix<Tm> get_Smat(const comb<Qm,Tm>& icomb){ 
         // loop over sites on backbone
         const auto& nodes = icomb.topo.nodes;
         const auto& rindex = icomb.topo.rindex;
         qtensor2<Qm::ifabelian,Tm> qt2_r, qt2_u;
         for(int i=icomb.topo.nbackbone-1; i>0; i--){
            const auto& node = nodes[i][0];
            int tp = node.type;
            if(tp == 0 || tp == 1){
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               if(i == icomb.topo.nbackbone-1){
                  qt2_r = contract_qt3_qt3("cr",site,site);
               }else{
                  auto qtmp = contract_qt3_qt2("r",site,qt2_r);
                  qt2_r = contract_qt3_qt3("cr",site,qtmp);
               }
            }else if(tp == 3){
               for(int j=nodes[i].size()-1; j>=1; j--){
                  const auto& site = icomb.sites[rindex.at(std::make_pair(i,j))];
                  if(j == nodes[i].size()-1){
                     qt2_u = contract_qt3_qt3("cr",site,site);
                  }else{
                     auto qtmp = contract_qt3_qt2("r",site,qt2_u);
                     qt2_u = contract_qt3_qt3("cr",site,qtmp);
                  }
               } // j
               // internal site without physical index
               const auto& site = icomb.sites[rindex.at(std::make_pair(i,0))];
               auto qtmp = contract_qt3_qt2("r",site,qt2_r); // ket
               qtmp = contract_qt3_qt2("c",qtmp,qt2_u); // upper branch
               qt2_r = contract_qt3_qt3("cr",site,qtmp); // bra
            }
         } // i
         // first merge: sum_l rwfuns[j,l]*site0[l,r,n] => site[j,r,n]
         const auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))];
         auto site = contract_qt3_qt2("l",site0,icomb.get_wf2());
         auto qtmp = contract_qt3_qt2("r",site,qt2_r);
         qt2_r = contract_qt3_qt3("cr",site,qtmp);
         auto Smat = qt2_r.to_matrix();
         return Smat;
      }

} // ctns

#endif
