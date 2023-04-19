#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "sweep_onedot_decimation.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <typename Km>
      void onedot_renorm(comb<Km>& icomb,
            const integral::two_body<typename Km::dtype>& int2e, 
            const integral::one_body<typename Km::dtype>& int1e,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<typename Km::dtype>& vsol,
            stensor3<typename Km::dtype>& wf,
            oper_pool<typename Km::dtype>& qops_pool,
            const std::vector<std::string>& fneed,
            const std::vector<std::string>& fneed_next,
            const std::string& frop,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         using Tm = typename Km::dtype;
         const bool ifkr = Km::ifkr;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool debug = (rank == 0);
         const auto& dbond = sweeps.seq[ibond];
         std::string superblock;
         if(dbond.forward){
            superblock = dbond.is_cturn()? "lr" : "lc";
         }else{
            superblock = "cr";
         }
         if(debug && schd.ctns.verbose>0){ 
            std::cout << "ctns::onedot_renorm superblock=" << superblock;
         }
         auto& timing = sweeps.opt_timing[isweep][ibond];

         // 1. build reduced density matrix & perform decimation
         stensor2<Tm> rot;
         if(rank == 0){
            auto dims = icomb.topo.check_partition(1, dbond, false);
            int ksupp;
            if(superblock == "lc"){
               ksupp = dims[0] + dims[2];
            }else if(superblock == "lr"){
               ksupp = dims[0] + dims[1];
            }else if(superblock == "cr"){
               ksupp = dims[1] + dims[2];
            }
            std::string fname = scratch+"/decimation"
               + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond)+".txt";
            onedot_decimation(schd, sweeps, isweep, ibond, ifkr, 
                  superblock, ksupp, vsol, wf, rot, fname);
         }
#ifndef SERIAL
         if(size > 1) mpi_wrapper::broadcast(icomb.world, rot, 0); 
#endif
         timing.td = tools::get_time();

         // 2. prepare guess for the next site
         if(rank == 0 && schd.ctns.guess){
            onedot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
         }
         vsol.clear();
         timing.te = tools::get_time();

         // 3. renorm operators	 
         auto& qops  = qops_pool(frop);
         auto& lqops = qops_pool(fneed[0]);
         auto& rqops = qops_pool(fneed[1]);
         auto& cqops = qops_pool(fneed[2]);
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p); 
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/rformulae"
            + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         if(superblock == "lc"){
            icomb.sites[pdx] = rot.split_lc(wf.info.qrow, wf.info.qmid);
            //-------------------------------------------------------------------
            if(check_canon){
               rot -= icomb.sites[pdx].merge_lc();
               assert(rot.normF() < thresh_canon);
               auto ovlp = contract_qt3_qt3("lc", icomb.sites[pdx], icomb.sites[pdx]);
               assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
            }
            //-------------------------------------------------------------------
            qops_pool.release({fneed[1]}, fneed_next);
            oper_renorm("lc", icomb, p, int2e, int1e, schd,
                  lqops, cqops, qops, fname, timing);
            qops_pool.release({fneed[0],fneed[2]}, fneed_next); 
         }else if(superblock == "lr"){
            icomb.sites[pdx]= rot.split_lr(wf.info.qrow, wf.info.qcol);
            //-------------------------------------------------------------------
            if(check_canon){
               rot -= icomb.sites[pdx].merge_lr();
               assert(rot.normF() < thresh_canon);
               auto ovlp = contract_qt3_qt3("lr", icomb.sites[pdx],icomb.sites[pdx]);
               assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
            }
            //-------------------------------------------------------------------
            qops_pool.release({fneed[2]}, fneed_next);
            oper_renorm("lr", icomb, p, int2e, int1e, schd,
                  lqops, rqops, qops, fname, timing); 
            qops_pool.release({fneed[0],fneed[1]}, fneed_next); 
         }else if(superblock == "cr"){
            icomb.sites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qcol);
            //-------------------------------------------------------------------
            if(check_canon){
               rot -= icomb.sites[pdx].merge_cr();
               assert(rot.normF() < thresh_canon);
               auto ovlp = contract_qt3_qt3("cr", icomb.sites[pdx],icomb.sites[pdx]);
               assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
            }
            //-------------------------------------------------------------------
            qops_pool.release({fneed[0]}, fneed_next);
            oper_renorm("cr", icomb, p, int2e, int1e, schd,
                  cqops, rqops, qops, fname, timing); 
            qops_pool.release({fneed[2],fneed[1]}, fneed_next); 
         }
      }

} // ctns

#endif
