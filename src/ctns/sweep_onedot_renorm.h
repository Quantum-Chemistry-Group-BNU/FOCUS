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
            const oper_dictmap<typename Km::dtype>& qops_dict,
            oper_dict<typename Km::dtype>& qops,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
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
         auto& CPUmem = sweeps.opt_CPUmem[isweep][ibond];

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
         if(debug){
            CPUmem.dvdson = 0;
            CPUmem.display();
         }
         timing.te = tools::get_time();

         // 3. renorm operators	 
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p); 
         std::string fname;
         if(schd.ctns.save_formulae) fname = scratch+"/rformulae"
            + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond) + ".txt";
         size_t worktot = 0;
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
            worktot = oper_renorm_opAll("lc", icomb, p, int2e, int1e, schd,
                  lqops, cqops, qops, fname, timing); 
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
            worktot = oper_renorm_opAll("lr", icomb, p, int2e, int1e, schd,
                  lqops, rqops, qops, fname, timing); 
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
            worktot = oper_renorm_opAll("cr", icomb, p, int2e, int1e, schd,
                  cqops, rqops, qops, fname, timing); 
         }
         if(debug){
            CPUmem.renorm = worktot;
            CPUmem.display();
         }
         timing.tf = tools::get_time();

         // save for restart
         if(rank == 0){
            std::string fsite = scratch+"/site_ibond"+std::to_string(ibond)+".info";
            rcanon_save(icomb.sites[pdx], fsite);
            if(schd.ctns.guess){ 
               std::string fcpsi = scratch+"/cpsi_ibond"+std::to_string(ibond)+".info";
               rcanon_save(icomb.cpsi, fcpsi);
            }
         } // only rank-0 save and load, later broadcast
      }

} // ctns

#endif
