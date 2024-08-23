#ifndef CTNS_DET2MPS_H
#define CTNS_DET2MPS_H

#include "../core/onstate.h"
#include "ctns_rcanon.h"

namespace ctns{

   // 000222 [little endian]
   // 543210
   template <typename Qm, typename Tm>
      comb<Qm,Tm> det2mps(const topology& topo,
            const fock::onstate& det,
            const bool ifrcanon=true){
         assert(Qm::ifabelian);
         int ks = det.norb();
         assert(topo.nphysical == ks);
         comb<Qm,Tm> mps;
         mps.topo = topo;
         mps.sites.resize(ks);
         auto narray  = det.intermediate_narray();
         auto tmarray = det.intermediate_tmarray();
         /*
            tools::print_vector(narray, "narray");
            tools::print_vector(tmarray, "tmarray");
            */
         int isym = Qm::isym;
         auto qmid = get_qbond_phys(isym);
         // loop from the last site to generate the RCF
         for(int i=ks-1; i>=0; i--){
            // l--<--*--<--r
            //       |
            //       phys
            qsym syml(isym,narray[i],tmarray[i]);
            qsym symr(isym,narray[i+1],tmarray[i+1]);
            qbond ql({{syml,1}}), qr({{symr,1}});
            qtensor3<Qm::ifabelian,Tm> site(qsym(isym,0,0),ql,qr,qmid,dir_RCF);
            assert(site.size() == 1);
            site._data[0] = 1.0;  
            mps.sites[ks-1-i] = std::move(site);
         }
         // rwfuns
         qbond qr({{qsym(isym,narray[0],tmarray[0]),1}});
         qtensor2<Qm::ifabelian,Tm> qt2(qsym(isym,0,0),qr,qr,dir_RWF);
         assert(qt2.size() == 1);
         qt2._data[0] = 1.0;
         mps.rwfuns.resize(1);
         mps.rwfuns[0] = std::move(qt2);

         // d=1 to rcanon for Hmat & DMRG
         if(ifrcanon) rcanon_lastdots(mps);
         /*
         // check
         rcanon_Sdiag_sample(mps, 0, 100, -1);
         */
         return mps;
      }

} // ctns

#endif
