#ifndef SWEEP_ONEDOT_RENORM_H
#define SWEEP_ONEDOT_RENORM_H

#include "sweep_onedot_decim.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   const bool check_canon = false;
   extern const bool check_canon;

   const double thresh_canon = 1.e-10;
   extern const double thresh_canon;

   template <typename Qm, typename Tm>
      void onedot_renorm(comb<Qm,Tm>& icomb,
            const integral::two_body<Tm>& int2e, 
            const integral::one_body<Tm>& int1e,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& vsol,
            stensor3<Tm>& wf,
            oper_pool<Tm>& qops_pool,
            const std::vector<std::string>& fneed,
            const std::vector<std::string>& fneed_next,
            const std::string& frop,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
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
         auto& timing = sweeps.opt_timing[isweep][ibond];

         // 1. build reduced density matrix & perform decimation
         stensor2<Tm> rot;
         onedot_decimation(icomb, schd, scratch, sweeps, isweep, ibond, 
               superblock, vsol, wf, rot);
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
         auto& qops  = qops_pool[frop];
         const auto& lqops = qops_pool.at(fneed[0]);
         const auto& rqops = qops_pool.at(fneed[1]);
         const auto& cqops = qops_pool.at(fneed[2]);
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p); 
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/rformulae"
            + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         std::string fmmtask;
         if(debug && schd.ctns.save_mmtask && isweep == schd.ctns.maxsweep-1 && ibond==schd.ctns.maxbond){
            fmmtask = "rmmtasks_isweep"+std::to_string(isweep) + "_ibond"+std::to_string(ibond);
         }
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
            qops_pool.clear_from_memory({fneed[1]}, fneed_next);
            oper_renorm("lc", icomb, p, int2e, int1e, schd,
                  lqops, cqops, qops, fname, timing, fmmtask);
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
            qops_pool.clear_from_memory({fneed[2]}, fneed_next);
            oper_renorm("lr", icomb, p, int2e, int1e, schd,
                  lqops, rqops, qops, fname, timing, fmmtask); 
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
            qops_pool.clear_from_memory({fneed[0]}, fneed_next);
            oper_renorm("cr", icomb, p, int2e, int1e, schd,
                  cqops, rqops, qops, fname, timing, fmmtask); 
         }
      }

} // ctns

#endif
