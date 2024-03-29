#ifndef CTNS_TOSU2_H
#define CTNS_TOSU2_H

#include "../init_phys.h"
#include "ctns_tosu2_qbond3.h"
#include "ctns_tosu2_wmat.h"
#include "ctns_tosu2_msite.h"
#include "ctns_tosu2_csite.h"
#include "ctns_tosu2_dm.h"
#include "ctns_tosu2_env.h"
#include "ctns_tosu2_update.h"
#include "ctns_tononsu2.h"

#include "../../core/csf.h"

namespace ctns{

   // convert to SU2 symmetry via sweep projection
   template <typename Tm>
      void rcanon_tosu2(const comb<qkind::qNSz,Tm>& icomb_NSz,
            comb<qkind::qNS,Tm>& icomb,
            const int twos,
            const double thresh_tosu2){
         const bool debug = false;
         std::cout << "\nctns::rcanon_tosu2 twos=" << twos 
            << " thresh_tosu2=" << thresh_tosu2
            << std::endl;
         auto t0 = tools::get_time();

         // build environment
         auto dmenv = buildDMLeftEnv(icomb_NSz, debug);
         
         // sweep projection: start from the last site
         icomb.topo = icomb_NSz.topo;
         int nsite = icomb_NSz.get_nphysical();
         icomb.sites.resize(nsite);

         // the right most site is identity, to be consistent with 
         // the initialization of boundary operators in oper_env.h
         icomb.sites[0] = get_right_bsite<qkind::qNS,Tm>();
         
         // initial Wmatrix
         Wmatrix<Tm> wmat;
         wmat = initW0site0<Tm>();

         for(int i=1; i<nsite; i++){

            std::cout << "idx=" << i << std::endl;
            if(debug) icomb_NSz.sites[i].print("rsite_"+std::to_string(i));

            // 1. form MixedRSite
            auto msite = formMixedRSite(icomb_NSz.sites[i], wmat, debug);

            // 2. form CoupledRSite [MOST IMPORTANT STEP!]
            const auto& qc = msite.qmid;
            const auto& qr = msite.qcol;
            auto qprod = qmerge(qc,qr);
            auto csite = formCoupledRSite(msite, qprod, qc, qr, debug);

            // 3. density matrix
            auto cdm = formCoupledDM(csite, dmenv[i], debug);

            // 4. decimation by diagonlizing quasi-dm 
            auto Yinfo = decimQuasiDM(cdm, thresh_tosu2, debug);

            // 5. update information: W
            wmat = updateWmat(csite, Yinfo, debug);

            // 6. expand Y into sa-mps site
            icomb.sites[i] = updateSite(Yinfo, qprod, qc, qr, debug);

            // debug:
            auto qt2 = contract_qt3_qt3("cr", icomb.sites[i], icomb.sites[i]);
            qt2.check_identityMatrix(1.e-10, false);
            if(debug) qt2.to_matrix().print("qt2mat");

         }

         if(debug){
            std::cout << "\n#######" << std::endl;
            std::cout << " FINAL" << std::endl;
            std::cout << "#######" << std::endl;
            wmat.qrow.print("qrow");
            display_qbond3(wmat.qcol,"qcol");
         }

         // form rwfuns
         icomb.rwfuns = updateRWFuns(icomb_NSz, wmat, twos);

         std::cout << "\nSummary of sweep projection: nroot=" << icomb_NSz.rwfuns.size()
            << " final nstate=" << icomb.rwfuns.size()
            << std::endl;

         icomb_NSz.display_shape();
         icomb.display_shape();

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_tosu2", t0, t1);
      }

} // ctns

#endif
