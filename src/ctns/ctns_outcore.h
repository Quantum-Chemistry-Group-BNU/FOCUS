#ifndef CTNS_OUTCORE_H
#define CTNS_OUTCORE_H

#include "../core/serialization.h"
#include "../core/mem_status.h"

namespace ctns{

   // set the site to an empty tensor to save memory;
   template <typename Qm, typename Tm>
      void rcanon_clear_site(comb<Qm,Tm>& icomb,
            const int idx){
         icomb.sites[idx] = std::move(qtensor3<Qm::ifabelian,Tm>());
      }

   template <typename Qm, typename Tm>
      void rcanon_save_sites(comb<Qm,Tm>& icomb,
            const std::string scratch,
            const int rank){
         if(rank == 0){
            std::cout << "ctns::rcanon_save_sites scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         // save sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto fname = scratch+"/site"+std::to_string(idx)+".temp";
            icomb.sites[idx].save_site(fname);
            rcanon_clear_site(icomb, idx);
         }
#ifdef TCMALLOC
         release_freecpumem();
#endif
         if(rank == 0){
            auto t1 = tools::get_time();
            double dt = tools::get_duration(t1-t0);
            std::cout << std::setprecision(2) << " T(save)=" << dt << "S" << std::endl; 
         }
      }
   template <typename Qm, typename Tm>
      void rcanon_load_sites(comb<Qm,Tm>& icomb,
            const std::string scratch,
            const int rank){
         if(rank == 0){
            std::cout << "ctns::rcanon_load_sites scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         // load sites
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            auto fname = scratch+"/site"+std::to_string(idx)+".temp";
            icomb.sites[idx].load_site(fname);
         }
         if(rank == 0){
            auto t1 = tools::get_time();
            double dt = tools::get_duration(t1-t0);
            std::cout << std::setprecision(2) << " T(load)=" << dt << "S" << std::endl;
         } 
      }

   template <typename Qm, typename Tm>
      void rcanon_save_site(comb<Qm,Tm>& icomb,
            const int idx,
            const std::string scratch,
            const int rank){
         if(rank == 0){
            std::cout << "ctns::rcanon_save_site idx=" << idx << " scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         size_t sz = icomb.sites[idx].size();
         // save site
         auto fname = scratch+"/site"+std::to_string(idx)+".temp";
         icomb.sites[idx].save_site(fname);
         // clear
         rcanon_clear_site(icomb, idx);
         if(rank == 0){
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
            const int rank){
         if(rank == 0){
            std::cout << "ctns::rcanon_load_site idx=" << idx << " scratch=" << scratch;
         }
         auto t0 = tools::get_time();
         // load site
         auto fname = scratch+"/site"+std::to_string(idx)+".temp";
         icomb.sites[idx].load_site(fname);
         if(rank == 0){
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
