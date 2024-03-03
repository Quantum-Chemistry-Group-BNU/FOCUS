#ifndef CTNS_CSF2SAMPS
#define CTNS_CSF2SAMPS

#include "../../core/csf.h"

namespace ctns{

   template <typename Tm>
      comb<qkind::qNS,Tm> csf2samps(const topology& topo,
            const fock::csfstate& csf){
         int ks = csf.norb();
         assert(topo.nphysical == ks);
         comb<qkind::qNS,Tm> samps;
         samps.topo = topo;
         samps.sites.resize(ks);
         auto narray  = csf.intermediate_narray();
         auto tsarray = csf.intermediate_tsarray();
         /*
         std::cout << "csf=" << csf << std::endl;
         tools::print_vector(narray, "narray");
         tools::print_vector(tsarray, "tsarray");
         */
         int isym = 3;
         auto qmid = get_qbond_phys(isym);
         // loop from the last site to generate the RCF
         for(int i=ks-1; i>=0; i--){
            // l--<--*--<--r
            //       |
            //       phys
            qsym syml(isym,narray[i],tsarray[i]);
            qsym symr(isym,narray[i+1],tsarray[i+1]);
            qbond ql({{syml,1}}), qr({{symr,1}});
            stensor3su2<Tm> site(qsym(isym,0,0),ql,qr,qmid,dir_RCF,CRcouple);
            assert(site.size() == 1);
            site._data[0] = 1.0;  
            samps.sites[ks-1-i] = std::move(site);
         }
         // rwfuns
         qbond qr({{qsym(isym,narray[0],tsarray[0]),1}});
         stensor2su2<Tm> qt2(qsym(isym,0,0),qr,qr,dir_RWF);
         assert(qt2.size() == 1);
         qt2._data[0] = 1.0;
         samps.rwfuns.resize(1);
         samps.rwfuns[0] = std::move(qt2);
         return samps;
      }

} // ctns

#endif
