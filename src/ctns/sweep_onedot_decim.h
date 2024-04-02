#ifndef SWEEP_ONEDOT_DECIM_H
#define SWEEP_ONEDOT_DECIM_H

#include "oper_io.h"
#include "sweep_decim.h"

namespace ctns{

   const double thresh_noise = 1.e-10;
   extern const double thresh_noise;

   // do not perform truncation at the boundary
   inline bool start_truncation(const int ksupp, 
         const int dcut){
      int knotrunc = std::min(4,(int)(0.5*std::log2(dcut)));
      return ksupp > knotrunc;
   }

   template <typename Qm, typename Tm>
      void onedot_decimation(const comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            sweep_data& sweeps,
            const int isweep,
            const int ibond, 
            const std::string superblock,
            const linalg::matrix<Tm>& vsol,
            stensor3<Tm>& wf,
            stensor2<Tm>& rot){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
         const bool debug = (rank==0 && schd.ctns.verbose>0);
         std::string fname = scratch+"/decimation"
            + "_isweep"+std::to_string(isweep)
            + "_ibond"+std::to_string(ibond)+".txt";
         const auto& dbond = sweeps.seq[ibond];
         auto dims = icomb.topo.check_partition(1, dbond, false);
         int ksupp;
         if(superblock == "lc"){
            ksupp = dims[0] + dims[2];
         }else if(superblock == "lr"){
            ksupp = dims[0] + dims[1];
         }else if(superblock == "cr"){
            ksupp = dims[1] + dims[2];
         }
         const auto& rdm_svd = schd.ctns.rdm_svd;
         const int& dbranch = schd.ctns.dbranch;
         const int dcut = (dbranch>0 && dbond.p1.second>0)? dbranch : sweeps.ctrls[isweep].dcut;
         const bool iftrunc = start_truncation(ksupp, dcut) && !schd.ctns.notrunc;
         const auto& noise = sweeps.ctrls[isweep].noise;
         if(debug){
            std::cout << "ctns::onedot_renorm superblock=" << superblock
               << " (rdm_svd,dbranch,dcut,iftrunc,noise)=" 
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
            decimation_row(icomb, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
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
            decimation_row(icomb, wf.info.qrow, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "cr"){

            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
               // wf3[l,r,c] => wf2[l,cr]
               auto wf2 = wf.merge_cr().P();
               if(noise > thresh_noise) wf2.add_noise(noise);
               wfs2[i] = std::move(wf2);
            }
            decimation_row(icomb, wf.info.qmid, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.P(); // rot[alpha,r] = (V^+)

         } // superblock
      }

} // ctns

#endif
