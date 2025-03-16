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

   //--- for restart ---
   template <typename Tm>
      void sweep_saveitem(const Tm& data,
            const std::string fname,
            const bool debug=true){
         if(debug) std::cout << "ctns::sweep_saveitem fname=" << fname << std::endl;
         std::ofstream ofs(fname+".info", std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         save << data;
         ofs.close();
      }

   template <typename Tm>
      void sweep_loaditem(Tm& data,
            const std::string fname,
            const bool debug=true){
         if(debug) std::cout << "ctns:sweep_loaditem fname=" << fname << std::endl;
         std::ifstream ifs(fname+".info", std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         load >> data;
         ifs.close();
      }

   // save for restart: only rank-0 save and load, later broadcast
   template <typename Qm, typename Tm>
      void sweep_save(const comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            const sweep_data& sweeps,
            const int isweep,
            const int ibond){
         std::cout << "ctns::sweep_save isweep=" << isweep << " ibond=" << ibond << std::endl;
         
         // local result
         std::string fresult = scratch+"/restart_result_ibond"+std::to_string(ibond);
         sweep_saveitem(sweeps.opt_result[isweep][ibond], fresult);
         
         // save site
         const auto& dbond = sweeps.seq[ibond];
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p); 
         auto fname = scratch+"/restart_site_ibond"+std::to_string(ibond)+".info";
         std::cout << "save_site fname=" << fname << std::endl;
         icomb.sites[pdx].save_site(fname);
         
         // generated cpsi
         if(schd.ctns.guess){ 
            std::string fcpsi = scratch+"/restart_cpsi_ibond"+std::to_string(ibond);
            sweep_saveitem(icomb.cpsi, fcpsi);
         }
      }

   template <typename Qm, typename Tm>
      void sweep_load(comb<Qm,Tm>& icomb,
            const input::schedule& schd,
            const std::string scratch,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         std::cout << "ctns::sweep_load isweep=" << isweep << " ibond=" << ibond << std::endl;
         auto t0 = tools::get_time();
         double sizeGB = 0.0;
         
         // load local result
         std::string fresult = scratch+"/restart_result_ibond"+std::to_string(ibond);
         sweep_loaditem(sweeps.opt_result[isweep][ibond], fresult);
         sweeps.opt_result[isweep][ibond].print();
         
         // load site
         const auto& dbond = sweeps.seq[ibond];
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p);
         auto fname = scratch+"/restart_site_ibond"+std::to_string(ibond)+".info";
         std::cout << "load site: fname=" << fname << std::endl;
         icomb.sites[pdx].load_site(fname);
         std::cout << " size(GB) of sites[pdx=" << pdx << "]="
            << tools::sizeGB<Tm>(icomb.sites[pdx].size()) 
            << std::endl;
         sizeGB += tools::sizeGB<Tm>(icomb.sites[pdx].size());
         
         // load cpsi
         if(schd.ctns.guess){ 
            std::string fcpsi = scratch+"/restart_cpsi_ibond"+std::to_string(ibond);
            icomb.cpsi.clear(); // to avoid memory leak
            sweep_loaditem(icomb.cpsi, fcpsi);
            for(int iroot=0; iroot<icomb.cpsi.size(); iroot++){
               std::cout << " size(GB) of cpsi[iroot=" << iroot << "]=" 
                  << tools::sizeGB<Tm>(icomb.cpsi[iroot].size()) 
                  << std::endl;
               sizeGB += tools::sizeGB<Tm>(icomb.cpsi[iroot].size()); 
            }
         }
         
         auto t1 = tools::get_time();
         double dt = tools::get_duration(t1-t0);
         std::cout << "----- TIMING FOR sweep_load : " << dt << " S"
           << " size(GB)=" << sizeGB << " speed=" << sizeGB/dt << "GB/S -----"
           << std::endl; 
      }

} // ctns

#endif
