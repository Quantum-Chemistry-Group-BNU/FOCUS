#ifndef SWEEP_UTIL_H
#define SWEEP_UTIL_H

namespace ctns{

   // look ahead for the next fneed
   template <typename Qm, typename Tm>
      std::vector<std::string> sweep_fneed_next(const comb<Qm,Tm>& icomb,
            const std::string scratch,
            const sweep_data& sweeps,
            const int isweep,
            const int ibond,
            const bool debug){
         const auto& dots = sweeps.ctrls[isweep].dots;
         std::vector<std::string> fneed_next;
         if(ibond != sweeps.seqsize-1){
            const auto& dbond_next = sweeps.seq[ibond+1]; 
            fneed_next = icomb.topo.get_fqops(dots, dbond_next, scratch, debug);
         }else{
            if(isweep != sweeps.maxsweep-1){
               const auto& dbond_next = sweeps.seq[0];
               const auto& dots = sweeps.ctrls[isweep+1].dots; 
               fneed_next = icomb.topo.get_fqops(dots, dbond_next, scratch);
            }
         }
         return fneed_next;
      }

   // save for restart: only rank-0 save and load, later broadcast
   template <typename Qm, typename Tm>
      void sweep_save(const comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            const sweep_data& sweeps,
            const int isweep,
            const int ibond){
         // local result
         std::string fresult = scratch+"/result_ibond"+std::to_string(ibond);
         rcanon_save(sweeps.opt_result[isweep][ibond], fresult);
         // updated site
         std::string fsite = scratch+"/site_ibond"+std::to_string(ibond);
         const auto& dbond = sweeps.seq[ibond];
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p); 
         rcanon_save(icomb.sites[pdx], fsite);
         // generated cpsi
         if(schd.ctns.guess){ 
            std::string fcpsi = scratch+"/cpsi_ibond"+std::to_string(ibond);
            rcanon_save(icomb.cpsi, fcpsi);
         }
      }

   template <typename Qm, typename Tm>
      void sweep_load(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         // load local result
         std::string fresult = scratch+"/result_ibond"+std::to_string(ibond);
         rcanon_load(sweeps.opt_result[isweep][ibond], fresult);
         sweeps.opt_result[isweep][ibond].print();
         // load site
         std::string fsite = scratch+"/site_ibond"+std::to_string(ibond);
         const auto& dbond = sweeps.seq[ibond];
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p);
         rcanon_load(icomb.sites[pdx], fsite);
         // load cpsi
         if(schd.ctns.guess){ 
            std::string fcpsi = scratch+"/cpsi_ibond"+std::to_string(ibond);
            rcanon_load(icomb.cpsi, fcpsi);
         }
      }

} // ctns

#endif
