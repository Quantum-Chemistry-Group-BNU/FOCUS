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

namespace ctns{

   // convert to SU2 symmetry via sweep projection
   template <typename Tm>
      void rcanon_tosu2(const comb<qkind::qNSz,Tm>& icomb_NSz,
            comb<qkind::qNS,Tm>& icomb,
            const int twos,
            const double thresh_tosu2){
         const bool debug = true;
         std::cout << "\nctns::rcanon_tosu2 twos=" << twos 
            << " thresh_tosu2=" << thresh_tosu2
            << std::endl;
         auto t0 = tools::get_time();

         // build environment
         auto dmenv = buildDMLeftEnv(icomb_NSz);

         /*
         // debug
         rcanon_Sdiag_sample(icomb_NSz, 0, 100, 10);
         auto tmp = icomb_NSz.sites[0].merge_cr();
         tmp.print("tmp");
         tmp.to_matrix().print("tmp");
         auto tmp2 = dmenv[0].dot(tmp);
         auto rho = tmp.H().dot(tmp2);
         rho.print("rho");
         rho.to_matrix().print("rho");
         exit(1);
         */

         // initial Wmatrix
         Wmatrix<Tm> wmat;
         wmat = initW0vac<Tm>();

         // sweep projection: start from the last site
         int nsite = icomb_NSz.get_nphysical();
         for(int i=0; i<nsite; i++){

            // load site
            if(debug){
               std::cout << "\n######" << std::endl;
               std::cout << " i=" << i << std::endl;
               std::cout << "######" << std::endl;
               icomb_NSz.sites[i].print("rsite_"+std::to_string(i));
            }

            std::cout << "\nwmat" << std::endl;
            wmat.qrow.print("qrow");
            display_qbond3(wmat.qcol,"qcol");
            wmat.to_matrix().print("wmat");

            auto w = wmat.to_matrix();
            w.print("w");
            auto wwh = linalg::xgemm("N","N",w,w.H());
            wwh.print("wwh");
            wwh = wwh - linalg::identity_matrix<Tm>(w.rows());
            double diffw = wwh.normF();
            std::cout << "diffw=" << diffw << std::endl;
            //if(diffw > 1.e-10) exit(1);

            // form MixedRSite
            auto msite = formMixedRSite(icomb_NSz.sites[i], wmat);
            const auto& qc = msite.qmid;
            const auto& qr = msite.qcol;

            auto rmat = contract_qt3_qt3("cr",icomb_NSz.sites[i],icomb_NSz.sites[i]).to_matrix();
            rmat.print("rmat");

            auto rmat2 = contract_qt3_qt3_cr(msite,msite).to_matrix();
            rmat2.print("rmat2");

            auto dev2 = rmat2-rmat;
            auto diff2 = dev2.normF();
            std::cout << "diffRmat=" << diff2 << std::endl;
            //if(diff2 > 1.e-10) exit(1);


            // form CoupledRSite [!!!]
            auto csite = formCoupledRSite(msite);



            std::cout << "\ncsite" << std::endl;
            csite.qrow.print("qrow");
            display_qbond3(csite.qcol,"qcol");
            auto cmat = csite.to_matrix();
            cmat.print("csite");
            auto renv = linalg::xgemm("N","N",cmat,cmat.H());
            renv.print("renv");
            auto dev = rmat-renv;
            auto diff = dev.normF();
            std::cout << "diffRenv=" << diff << std::endl;
            //if(diff > 1.e-10) exit(1);

            std::cout << "\nDM:" << std::endl;
            dmenv[i].print("dmenv");
            dmenv[i].to_matrix().print("dmenv");
            double tr = std::real(dmenv[i].to_matrix().trace());
            std::cout << "tr(DMenv)=" << tr << std::endl;

            // density matrix
            auto cdm = formCoupledDM(csite, dmenv[i]);
            
            //if(i==1) exit(1);

            // decimation by diagonlizing quasi-dm 
            auto Yinfo = decimQuasiDM(cdm, thresh_tosu2);

 
            csite.to_matrix().print("csite");
            rmat.print("rmat");
            std::cout << "\nYinfo:" << std::endl;
            for(const auto& pr : Yinfo){
               std::cout << "sym=" << pr.first << std::endl;
               pr.second.print("Ymat");
            }


           // update information: W
            wmat = updateWmat(csite, Yinfo);



            // example Y into sa-mps site

            //if(i==0) exit(1);

         }

         std::cout << "\n### FINAL ###" << std::endl;
         wmat.qrow.print("qrow");
         display_qbond3(wmat.qcol,"qcol");

         // debug
         rcanon_Sdiag_sample(icomb_NSz, 0, 100, 10);

         finalWaveFunction(icomb_NSz.rwfuns, wmat);
         exit(1);

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_tosu2", t0, t1);
      }

} // ctns

#endif
