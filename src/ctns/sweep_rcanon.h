#ifndef SWEEP_RCANON_H
#define SWEEP_RCANON_H

#include "../core/tools.h"
#include "../core/linalg.h"

namespace ctns{

   // generate right canonical form (RCF) for later usage
   template <typename Km>
      void sweep_rcanon(comb<Km>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            const int isweep){
         using Tm = typename Km::dtype;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif 
         // LCRRR -> CRRRR
         if(rank == 0){
            auto rcanon_file = schd.scratch+"/rcanon_isweep"+std::to_string(isweep)+".info";
            std::cout << "ctns::sweep_rcanon: convert into RCF & save into "
               << rcanon_file << std::endl;
            std::cout << tools::line_separator << std::endl;
            // only rank-0 has psi from renorm
            const auto& wf = icomb.cpsi[0];
            // compute R1 from cpsi via decimation
            stensor2<Tm> rot;
            int nroots = schd.ctns.nroots;
            std::vector<stensor2<Tm>> wfs2(nroots);
            for(int i=0; i<nroots; i++){
               auto wf2 = icomb.cpsi[i].merge_cr().T();
               wfs2[i] = std::move(wf2);
            }
            const int dcut = 4*nroots; // psi[l,n,r,i] => U[l,i,a]sigma[a]Vt[a,n,r]
            double dwt; 
            int deff;
            const bool ifkr = Km::ifkr;
            std::string fname = "";
            const bool iftrunc = true;
            decimation_row(ifkr, wf.info.qmid, wf.info.qcol, 
                  iftrunc, dcut, schd.ctns.rdm_svd, wfs2,
                  rot, dwt, deff, fname, 
                  schd.ctns.verbose>0);
            rot = rot.T();
            const auto& pdx0 = icomb.topo.rindex.at(std::make_pair(0,0));
            const auto& pdx1 = icomb.topo.rindex.at(std::make_pair(1,0));
            icomb.sites[pdx1] = rot.split_cr(wf.info.qmid, wf.info.qcol);
            // compute C0 
            for(int i=0; i<nroots; i++){
               auto cwf = icomb.cpsi[i].merge_cr().dot(rot.H()); // <-W[l,alpha]->
               auto psi = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.T());
               icomb.cpsi[i] = std::move(psi);
            }
            icomb.stack_cpsi();
            ctns::rcanon_save(icomb, rcanon_file);
         } // rank0
      }

} // ctns

#endif
