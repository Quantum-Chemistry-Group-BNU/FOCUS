#ifndef SWEEP_TWODOT_RENORM_H
#define SWEEP_TWODOT_RENORM_H

#include "sweep_twodot_decimation.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <typename Km>
      void twodot_renorm(comb<Km>& icomb,
            const integral::two_body<typename Km::dtype>& int2e, 
            const integral::one_body<typename Km::dtype>& int1e,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<typename Km::dtype>& vsol,
            stensor4<typename Km::dtype>& wf,
            const oper_dictmap<typename Km::dtype>& qops_dict,
            oper_dict<typename Km::dtype>& qops,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
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
            superblock = dbond.is_cturn()? "lr" : "lc1";
         }else{
            superblock = dbond.is_cturn()? "c1c2" : "c2r";
         }
         if(debug && schd.ctns.verbose>0){ 
            std::cout << "ctns::twodot_renorm superblock=" << superblock;
         }
         auto& timing = sweeps.opt_timing[isweep][ibond];
         auto& CPUmem = sweeps.opt_CPUmem[isweep][ibond];

         // 1. build reduced density matrix & perform decimation
         stensor2<Tm> rot;
         if(rank == 0){
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
            std::string fname = scratch+"/decimation"
               + "_isweep"+std::to_string(isweep)
               + "_ibond"+std::to_string(ibond)+".txt";
            twodot_decimation(schd, sweeps, isweep, ibond, ifkr, 
                  superblock, ksupp, vsol, wf, rot, fname);
         }
#ifndef SERIAL
         if(size > 1) mpi_wrapper::broadcast(icomb.world, rot, 0); 
#endif
         timing.td = tools::get_time();

         // 2. prepare guess for the next site
         if(rank == 0 && schd.ctns.guess){
            twodot_guess_psi(superblock, icomb, dbond, vsol, wf, rot);
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
         if(superblock == "lc1"){
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
                  lqops, c1qops, qops, fname, timing); 
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
         }else if(superblock == "c2r"){
            icomb.sites[pdx] = rot.split_cr(wf.info.qver, wf.info.qcol);
            //-------------------------------------------------------------------
            if(check_canon){
               rot -= icomb.sites[pdx].merge_cr();
               assert(rot.normF() < thresh_canon);
               auto ovlp = contract_qt3_qt3("cr", icomb.sites[pdx],icomb.sites[pdx]);
               assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
            }
            //-------------------------------------------------------------------
            worktot = oper_renorm_opAll("cr", icomb, p, int2e, int1e, schd,
                  c2qops, rqops, qops, fname, timing);
         }else if(superblock == "c1c2"){
            icomb.sites[pdx] = rot.split_cr(wf.info.qmid, wf.info.qver);
            //-------------------------------------------------------------------
            if(check_canon){
               rot -= icomb.sites[pdx].merge_cr();
               assert(rot.normF() < thresh_canon);
               auto ovlp = contract_qt3_qt3("cr", icomb.sites[pdx],icomb.sites[pdx]);
               assert(ovlp.check_identityMatrix(thresh_canon) < thresh_canon);
            }
            //-------------------------------------------------------------------
            worktot = oper_renorm_opAll("cr", icomb, p, int2e, int1e, schd,
                  c1qops, c2qops, qops, fname, timing); 
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
