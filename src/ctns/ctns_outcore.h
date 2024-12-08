#ifndef CTNS_OUTCORE_H
#define CTNS_OUTCORE_H

#include "../core/serialization.h"

namespace ctns{

   // outcore
   template <typename Qm, typename Tm>
      void rcanon_clear_site(comb<Qm,Tm>& icomb,
            const int idx){
         // set the site to an empty tensor to save memory;
         icomb.sites[idx] = qtensor3<Qm::ifabelian,Tm>();
      }

   template <typename Qm, typename Tm>
      void rcanon_save_sites(comb<Qm,Tm>& icomb,
            const std::string scratch,
            const bool debug){
         if(debug){
            std::cout << "ctns::rcanon_save_sites scratch=" << scratch << std::endl;
         }
         // save sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            std::ofstream ofs(scratch+"/site"+std::to_string(idx)+".info", std::ios::binary);
            boost::archive::binary_oarchive save(ofs);
            save << icomb.sites[idx];;
            ofs.close();
            // clear
            rcanon_clear_site(icomb, idx);
         }
      }
   template <typename Qm, typename Tm>
      void rcanon_load_sites(comb<Qm,Tm>& icomb,
            const std::string scratch,
            const bool debug){
         if(debug){
            std::cout << "ctns::rcanon_load_sites scratch=" << scratch << std::endl;
         }
         // load sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            std::ifstream ifs(scratch+"/site"+std::to_string(idx)+".info", std::ios::binary);
            boost::archive::binary_iarchive load(ifs);
            load >> icomb.sites[idx];;
            ifs.close();
         }
      }

   template <typename Qm, typename Tm>
      void rcanon_save_site(comb<Qm,Tm>& icomb,
            const int idx,
            const std::string scratch,
            const bool debug){
         if(debug){
            std::cout << "ctns::rcanon_save_site idx=" << idx << " scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         size_t sz = icomb.sites[idx].size();
         // save site
         std::ofstream ofs(scratch+"/site"+std::to_string(idx)+".info", std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         save << icomb.sites[idx];;
         ofs.close();
         // clear
         rcanon_clear_site(icomb, idx);
         if(debug){
            auto t1 = tools::get_time();
            double dt = tools::get_duration(t1-t0);
            std::cout << " size=" << sz << ":" 
               << std::setprecision(2) << tools::sizeGB<Tm>(sz) << "GB" 
               << " T(load)=" << dt << "S"
               << " speed=" << tools::sizeGB<Tm>(sz)/dt << "GB/S"
               << std::endl; 
         }
      }
   template <typename Qm, typename Tm>
      void rcanon_load_site(comb<Qm,Tm>& icomb,
            const int idx,
            const std::string scratch,
            const bool debug){
         if(debug){
            std::cout << "ctns::rcanon_load_site idx=" << idx << " scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         // load site
         std::ifstream ifs(scratch+"/site"+std::to_string(idx)+".info", std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         load >> icomb.sites[idx];;
         ifs.close();
         if(debug){
            auto t1 = tools::get_time();
            size_t sz = icomb.sites[idx].size();
            double dt = tools::get_duration(t1-t0);
            std::cout << " size=" << sz << ":" 
               << std::setprecision(2) << tools::sizeGB<Tm>(sz) << "GB" 
               << " T(load)=" << dt << "S"
               << " speed=" << tools::sizeGB<Tm>(sz)/dt << "GB/S"
               << std::endl; 
         }
      }

} // ctns

#endif
