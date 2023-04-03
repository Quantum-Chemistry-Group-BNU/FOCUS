#ifndef SWEEP_TWODOT_DECIMATION_H
#define SWEEP_TWODOT_DECIMATION_H

#include "oper_io.h"
#include "sweep_decimation.h"
#include "sweep_onedot_renorm.h"

namespace ctns{

   template <typename Tm>
      void twodot_decimation(const input::schedule& schd,
            sweep_data& sweeps,
            const int isweep,
            const int ibond,
            const bool ifkr,
            const std::string superblock,
            const int ksupp,
            const linalg::matrix<Tm>& vsol, 
            stensor4<Tm>& wf,
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
         if(superblock == "lc1"){ 

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto wf2 = wf.merge_lc1_c2r();
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
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(ifkr, wf.info.qrow, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, wfs2, 
                  rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "c2r"){ 

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto wf2 = wf.merge_lc1_c2r().T();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(ifkr, wf.info.qver, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, wfs2, 
                  rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.T(); // rot[alpha,r] = (V^+)

         }else if(superblock == "c1c2"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2().T();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            } // i
            decimation_row(ifkr, wf.info.qmid, wf.info.qver, 
                  iftrunc, dcut, rdm_svd, wfs2,
                  rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.T(); // permute two lines for RCF

         } // superblock
      }

   // initial guess for next site within the bond
   template <typename Km>
      void twodot_guess_psi(const std::string superblock,
            comb<Km>& icomb,
            const directed_bond& dbond,
            const linalg::matrix<typename Km::dtype>& vsol,
            stensor4<typename Km::dtype>& wf,
            const stensor2<typename Km::dtype>& rot){
         const bool debug = false;
         if(debug) std::cout << "ctns::twodot_guess_psi superblock=" << superblock << std::endl;
         int nroots = vsol.cols();
         icomb.cpsi.clear();
         icomb.cpsi.resize(nroots);
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

         }else if(superblock == "lr"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               //-------------------------------------------
               // Two-dot case: simply use cwf[alpha,c2,c1]
               //-------------------------------------------
               // wf4[l,r,c1,c2] => wf2[lr,c1c2]
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2();
               // rot.H()[alpha,lr]*wf3[lr,c1c2] => cwf[alpha,c1c2]
               auto cwf = rot.H().dot(wf2);
               // cwf[alpha,c1c2] => cwf[alpha,c2,c1] 
               auto psi = cwf.split_cr(wf.info.qmid, wf.info.qver);
               //-------------------------------------------
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

         }else if(superblock == "c1c2"){

            for(int i=0; i<vsol.cols(); i++){
               wf.from_array(vsol.col(i));
               //----------------------------------------------
               // Two-dot case: simply use cwf[l,r,alpha]
               //----------------------------------------------
               wf.permCR_signed();
               // wf4[l,c1,c2,r] => wf2[lr,c1c2]
               auto wf2 = wf.merge_lr_c1c2();
               // wf2[lr,c1c2]*rot.H()[c1c2,alpha] => cwf[lr,alpha]
               auto cwf = wf2.dot(rot.H());
               // cwf[lr,alpha] => psi[l,r,alpha]
               auto psi = cwf.split_lr(wf.info.qrow, wf.info.qcol);
               // revert ordering of the underlying basis
               psi.permCR_signed(); 
               //----------------------------------------------
               icomb.cpsi[i] = std::move(psi); // psi on backbone
            }

         } // superblock
      }

} // ctns

#endif
