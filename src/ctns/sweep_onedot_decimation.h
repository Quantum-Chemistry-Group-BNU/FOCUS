#ifndef SWEEP_ONEDOT_DECIMATION_H
#define SWEEP_ONEDOT_DECIMATION_H

#include "oper_io.h"
#include "sweep_decimation.h"

namespace ctns{

   const double thresh_noise = 1.e-10;
   extern const double thresh_noise;

   const bool check_canon = false;
   extern const bool check_canon;

   const double thresh_canon = 1.e-10;
   extern const double thresh_canon;

   // do not perform truncation at the boundary
   inline bool start_truncation(const int ksupp, 
         const int dcut){
      int knotrunc = std::min(4,(int)(0.5*std::log2(dcut)));
      return ksupp > knotrunc;
   }

   template <typename Tm>
      void onedot_decimation(const input::schedule& schd,
            sweep_data& sweeps,
            const int isweep,
            const int ibond, 
            const bool ifkr,
            const std::string superblock,
            const int ksupp,
            const linalg::matrix<Tm>& vsol,
            stensor3<Tm>& wf,
            stensor2<Tm>& rot, 
            const std::string fname){
         const bool debug = schd.ctns.verbose>0;
         const auto& rdm_svd = schd.ctns.rdm_svd;
         const auto& dbond = sweeps.seq[ibond];
         const int& dbranch = schd.ctns.dbranch;
         const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
         const bool iftrunc = start_truncation(ksupp, dcut);
         const auto& noise = sweeps.ctrls[isweep].noise;
         if(debug){
            std::cout <<" (rdm_svd,dbranch,dcut,iftrunc,noise)=" 
               << std::scientific << std::setprecision(1) << rdm_svd << ","
               << dbranch << "," << dcut << "," << iftrunc << ","
               << noise << std::endl;
         }
         auto& result = sweeps.opt_result[isweep][ibond];
         int nroots = vsol.cols();
         std::vector<stensor2<Tm>> wfs2(nroots);
         if(superblock == "lc"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               // wf3[l,r,c] => wf2[lc,r]
               auto wf2 = wf.merge_lc();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(ifkr, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dcut, rdm_svd, wfs2, 
                  rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "lr"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               // Need to first bring two dimensions adjacent to each other before merge!
               wf.permCR_signed();
               // wf3[l,r,c] => wf2[lr,c]
               auto wf2 = wf.merge_lr();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(ifkr, wf.info.qrow, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, wfs2, 
                  rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "cr"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               // wf3[l,r,c] => wf2[l,cr]
               auto wf2 = wf.merge_cr().T();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(ifkr, wf.info.qmid, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, wfs2, 
                  rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.T(); // rot[alpha,r] = (V^+)

         } // superblock
      }

   // initial guess for next site within the bond
   template <typename Km>
      void onedot_guess_psi(const std::string superblock,
            comb<Km>& icomb,
            const directed_bond& dbond,
            const linalg::matrix<typename Km::dtype>& vsol,
            stensor3<typename Km::dtype>& wf,
            const stensor2<typename Km::dtype>& rot){
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
               auto cwf = rot.H().dot(wf.merge_lc()); // <-W[alpha,r]->
               auto psi = contract_qt3_qt2("l",icomb.sites[pdx1],cwf);
               icomb.cpsi[i] = std::move(psi);
            }

         }else if(superblock == "lr"){

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
               auto cwf = wf.merge_cr().dot(rot.H()); // <-W[l,alpha]->
               if(!cturn){
                  auto psi = contract_qt3_qt2("r",icomb.sites[pdx0],cwf.T());
                  icomb.cpsi[i] = std::move(psi);
               }else{
                  // special treatment of the propagation downside to backbone
                  auto psi = contract_qt3_qt2("c",icomb.sites[pdx0],cwf.T());
                  psi.permCR_signed(); // |(lr)c> back to |lcr> order on backbone
                  icomb.cpsi[i] = std::move(psi);
               }
            }

         } // superblock
      }

} // ctns

#endif
