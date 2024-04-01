#ifndef SWEEP_TWODOT_DECIM_SU2_H
#define SWEEP_TWODOT_DECIM_SU2_H

#include "../sweep_onedot_decim.h"
#include "sweep_decim_su2.h"

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
            stensor4su2<Tm>& wf,
            stensor2su2<Tm>& rot){
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
         const bool iftrunc = start_truncation(ksupp, dcut);
         const auto& noise = sweeps.ctrls[isweep].noise;
         if(debug){
            std::cout << "ctns::twodot_renorm(su2) superblock=" << superblock
               << " (rdm_svd,dbranch,dcut,iftrunc,noise)=" 
               << std::scientific << std::setprecision(1) << rdm_svd << ","
               << dbranch << "," << dcut << "," << iftrunc << ","
               << noise << std::endl;
         }
         auto& result = sweeps.opt_result[isweep][ibond];
         int nroots = vsol.cols();
         std::vector<stensor2su2<Tm>> wfs2(nroots);
         auto t1 = tools::get_time();
         if(superblock == "lc1"){ 

            wf.from_array(vsol.col(0));

            auto qs12 = wf.dpt_lc1().first;
            auto qs34 = wf.dpt_c2r().first;

            std::cout << "wfnorm=" << wf.normF() << std::endl;

            //stensor3su2<Tm> wf3a(wf.info.sym,qs12,wf.info.qcol,wf.info.qver,dir_WF3,CRcouple);
            //wf3a.print("wf3a");
          
/*
            auto wf3c = wf.merge_lc1();
            wf3c.print("wf3c");
            std::cout << "wf3norm=" << wf3c.normF() << std::endl;

            auto wf4 = wf3c.split_lc1(wf.info.qrow, wf.info.qmid);
            std::cout << "wf4norm=" << wf3c.normF() << std::endl;
            assert(wf.size() == wf4.size());
            wf4 -= wf;
            std::cout << "diff=" << wf4.normF() << std::endl;
*/

            auto wf3c = wf.merge_c2r();
            wf3c.print("wf3c");
            std::cout << "wf3norm=" << wf3c.normF() << std::endl;

            auto wf4 = wf3c.split_c2r(wf.info.qver, wf.info.qcol);
            std::cout << "wf4norm=" << wf3c.normF() << std::endl;
            assert(wf.size() == wf4.size());
            wf4 -= wf;
            std::cout << "diff=" << wf4.normF() << std::endl;

            auto wf2b = wf4.merge_lc1_c2r();
            wf2b.print("wf2b");

            auto wf2c = wf4.merge_c2r().merge_lc();
            wf2c -= wf2b;
            std::cout << "diff=" << wf2c.normF() << std::endl;
            exit(1);

            stensor3su2<Tm> wf3b(wf.info.sym,wf.info.qrow,qs34,wf.info.qmid,dir_WF3,LCcouple);
            wf3b.print("wf3b");

            stensor2su2<Tm> wf2(wf.info.sym,qs12,qs34,dir_WF2);
            wf2.print("wf2");

            std::cout << "HERE!" << std::endl;
            exit(1);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
//               auto wf2 = wf.merge_lc1_c2r();
//               if(noise > thresh_noise) wf2.add_noise(noise);
//               wfs2[i] = std::move(wf2);
            }
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qrow, wf.info.qmid, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);

         }else if(superblock == "c2r"){ 

            std::cout << "HERE!" << std::endl;
            exit(1);
            for(int i=0; i<nroots; i++){
               wf.from_array(vsol.col(i));
//               auto wf2 = wf.merge_lc1_c2r().P();
//               if(noise > thresh_noise) wf2.add_noise(noise);
//               wfs2[i] = std::move(wf2);
            }
            t1 = tools::get_time();
            decimation_row(icomb, wf.info.qver, wf.info.qcol, 
                  iftrunc, dcut, rdm_svd, schd.ctns.alg_decim,
                  wfs2, rot, result.dwt, result.deff, fname,
                  debug);
            rot = rot.P(); // rot[alpha,r] = (V^+)

         } // superblock
         if(debug){
            auto t2 = tools::get_time();
            std::cout << "TIMING FOR decimation: "
               << tools::get_duration(t2-t0) << " S"
               << " T(wf/decim)=" 
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1)
               << std::endl;
         }
      }

} // ctns

#endif
