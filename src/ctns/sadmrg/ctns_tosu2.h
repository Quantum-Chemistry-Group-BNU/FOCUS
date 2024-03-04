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

#include "../../core/csf.h"

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
         */

         // initial Wmatrix
         Wmatrix<Tm> wmat;
         wmat = initW0vac<Tm>();

         // sweep projection: start from the last site
         int nsite = icomb_NSz.get_nphysical();
         icomb.sites.resize(nsite);

         for(int i=0; i<nsite; i++){

            // load site
            if(debug){
               std::cout << "\n######" << std::endl;
               std::cout << " i=" << i << std::endl;
               std::cout << "######" << std::endl;
               icomb_NSz.sites[i].print("rsite_"+std::to_string(i));
            }

            /*
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
            */

            // 1. form MixedRSite
            auto msite = formMixedRSite(icomb_NSz.sites[i], wmat);

            /*
               auto rmat = contract_qt3_qt3("cr",icomb_NSz.sites[i],icomb_NSz.sites[i]).to_matrix();
               rmat.print("rmat");

               auto rmat2 = contract_qt3_qt3_cr(msite,msite).to_matrix();
               rmat2.print("rmat2");

               auto dev2 = rmat2-rmat;
               auto diff2 = dev2.normF();
               std::cout << "diffRmat=" << diff2 << std::endl;
            //if(diff2 > 1.e-10) exit(1);
            */

            // 2. form CoupledRSite [MOST IMPORTANT STEP!]
            const auto& qc = msite.qmid;
            const auto& qr = msite.qcol;
            auto qprod = qmerge(qc,qr);
            auto csite = formCoupledRSite(msite, qprod, qc, qr);

            /*
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
            */

            // 3. density matrix
            auto cdm = formCoupledDM(csite, dmenv[i]);

            // 4. decimation by diagonlizing quasi-dm 
            auto Yinfo = decimQuasiDM(cdm, thresh_tosu2);

            /* 
               csite.to_matrix().print("csite");
               rmat.print("rmat");
               std::cout << "\nYinfo:" << std::endl;
               for(const auto& pr : Yinfo){
               std::cout << "sym=" << pr.first << std::endl;
               pr.second.print("Ymat");
               }
               */

            // 5. update information: W
            wmat = updateWmat(csite, Yinfo);

            // 6. expand Y into sa-mps site
            icomb.sites[i] = updateSite(Yinfo, qprod, qc, qr);

            // debug:
            auto qt2 = contract_qt3_qt3("cr", icomb.sites[i], icomb.sites[i]);
            qt2.check_identityMatrix(1.e-10, false);
            qt2.to_matrix().print("qt2mat");

         }

         if(debug){
            std::cout << "\n#######" << std::endl;
            std::cout << " FINAL" << std::endl;
            std::cout << "#######" << std::endl;
            wmat.qrow.print("qrow");
            display_qbond3(wmat.qcol,"qcol");
         }
         // debug
         //rcanon_Sdiag_sample(icomb_NSz, 0, 100, 10);

         /*
            auto wf2 = icomb_NSz.get_wf2();
            wf2.to_matrix().print("wf2");
            auto ovlp = (wf2.dot(wf2.H())).conj();
            ovlp.to_matrix().print("ovlp");
            */
         icomb.rwfuns = updateRWFuns(icomb_NSz, wmat, twos);

         /*
            auto wf2 = icomb.get_wf2();
            wf2.to_matrix().print("wf2");
            auto ovlp = (wf2.dot(wf2.H())).conj();
            ovlp.to_matrix().print("ovlp");
            ovlp.check_identityMatrix(1.e-10);
            */

         std::cout << "\nSummary of sweep projection: nroot=" << icomb_NSz.rwfuns.size()
            << " final nstate=" << icomb.rwfuns.size()
            << std::endl;

         icomb_NSz.display_shape();
         icomb.display_shape();

         int iroot = 0;
         /*
         auto Sij = ctns::get_Smat(icomb);
         Sij.print("Sij");

         auto expansion = rcanon_expand_onstate(icomb_NSz, iroot);
         rcanon_Sdiag_exact(icomb_NSz, iroot);
         rcanon_Sdiag_sample(icomb_NSz, iroot);

         auto expansion1 = rcanon_expand_csfstate(icomb, iroot);
         auto expansion2 = rcanon_expand_onstate(icomb, iroot);
         auto ova = linalg::xdot(expansion2.first.size(), expansion.second.data(), expansion2.second.data());
         std::cout << "ova=" << ova << " p2=" << std::setprecision(10) << ova*ova << std::endl;
         */

         rcanon_Sdiag_exact(icomb, iroot, "csf");
         rcanon_Sdiag_exact(icomb, iroot, "det");
        
         for(int i=0; i<100; i++){ 
            rcanon_random(icomb, iroot, true);
         }
         rcanon_Sdiag_sample(icomb, iroot);
         exit(1);

         /*
            std::cout << "\nrcanon_CIcoeff:" << std::endl;
            fock::onspace fci_space;
            auto sym_state = icomb.get_sym_state();
            fci_space = fock::get_fci_space(2*nsite, sym_state.ne()); 
            size_t dim = fci_space.size(); 
            std::vector<Tm> v(dim);
            for(int i=0; i<dim; i++){
            auto coeff = rcanon_CIcoeff(icomb, fci_space[i]);
            std::cout << "i=" << i 
            << " state=" << fci_space[i]
            << " <n|CTNS[0]>=" << coeff[0] 
            << std::endl;   
            v[i] = coeff[0];
            }
            auto overlap = std::pow(linalg::xnrm2(dim, v.data()),2);
            std::cout << "<v|v>=" << overlap << std::endl;
            */

         auto t1 = tools::get_time();
         tools::timing("ctns::rcanon_tosu2", t0, t1);
      }

} // ctns

#endif
