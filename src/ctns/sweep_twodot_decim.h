#ifndef SWEEP_TWODOT_DECIM_H
#define SWEEP_TWODOT_DECIM_H

#include "sweep_onedot_decim.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void twodot_decimation(const comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            sweep_data& sweeps,
            const int isweep,
            const int ibond,
            const std::string superblock,
            const linalg::matrix<Tm>& vsol, 
            qtensor4<Qm::ifabelian,Tm>& wf,
            qtensor2<Qm::ifabelian,Tm>& rot){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         auto t0 = tools::get_time();
         const bool debug = (rank==0 && schd.ctns.verbose>0);
         std::string fname = scratch+"/decimation"
            + "_isweep"+std::to_string(isweep)
            + "_ibond"+std::to_string(ibond)+".txt";
         const auto& dbond = sweeps.seq[ibond];
         auto dims = icomb.topo.check_partition(2, dbond, false);
         int ksupp;
         if(superblock == "lc1"){
            ksupp = dims[0] + dims[2];
         }else if(superblock == "lr"){
            ksupp = dims[0] + dims[1];
         }else if(superblock == "c2r"){
            ksupp = dims[1] + dims[3];
         }else if(superblock == "c1c2"){
            ksupp = dims[2] + dims[3];
         }
         const auto& rdm_svd = schd.ctns.rdm_svd;
         const int& dbranch = schd.ctns.dbranch;
         const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
         const bool iftrunc = start_truncation(ksupp, dcut) && !schd.ctns.notrunc;
         const auto& noise = sweeps.ctrls[isweep].noise;
         if(debug){
            std::cout << "ctns::twodot_renorm superblock=" << superblock
               << " (rdm_svd,dbranch,dcut,iftrunc,noise)=" 
               << std::scientific << std::setprecision(1) << rdm_svd << ","
               << dbranch << "," << dcut << "," << iftrunc << ","
               << noise << std::endl;
         }
         auto& result = sweeps.opt_result[isweep][ibond];
         int nroots = vsol.cols();
         std::vector<qtensor2<Qm::ifabelian,Tm>> wfs2(nroots);
         auto t1 = tools::get_time();
         if(superblock == "lc1"){ 

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto wf2 = wf.merge_lc1_c2r();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "c2r"){ 

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               auto wf2 = wf.merge_lc1_c2r().P();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qver, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.P(); // rot[alpha,r] = (V^+)

         }else if(superblock == "lr"){ 

            assert(Qm::ifabelian);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qrow, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "c1c2"){

            assert(Qm::ifabelian);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               wf.permCR_signed();
               auto wf2 = wf.merge_lr_c1c2().P();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            } // i
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qmid, wf.info.qver, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.P(); // permute two lines for RCF

         } // superblock
         if(debug){
            auto t2 = tools::get_time();
            std::cout << "----- TIMING FOR twodot_decimation: "
               << tools::get_duration(t2-t0) << " S"
               << " T(wf/decim)=" 
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << " S -----"
               << std::endl;
         }
      }

} // ctns

#endif
