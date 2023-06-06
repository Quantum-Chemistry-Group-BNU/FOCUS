#ifndef OPER_POOL_RAW_H
#define OPER_POOL_RAW_H

#include <thread>
#include <memory>
#include "oper_dict.h"
#include "oper_io.h"

namespace ctns{

   // 
   // oper_pool_raw with std::map 
   //

   template <typename Tm>
      using operData_pool_raw = std::map<std::string,oper_dict<Tm>>;

   // pool for mananging operators
   template <typename Tm>
      struct oper_pool_raw{
         public:
            // constuctor
            oper_pool_raw(const int _iomode, const bool _debug): iomode(_iomode), debug(_debug){}
            // check whether frop exist in the pool
            bool exist(const std::string& frop) const{
               return qstore.find(frop) != qstore.end();
            }
            // access
            const oper_dict<Tm>& at(const std::string& fqop) const{
               assert(this->exist(fqop));
               return qstore.at(fqop);
            }
            oper_dict<Tm>& operator[](const std::string& fqop){
               return qstore[fqop];
            }
            // total size allocated for storing operators
            size_t size() const{
               size_t sz = 0;
               for(auto pr = qstore.cbegin(); pr != qstore.cend(); pr++){
                  sz += pr->second.size();
               }
               return sz;
            }
            void display(const std::string msg="") const{
               std::cout << "qstore";
               if(msg.size()>0) std::cout << "[" << msg << "]";
               std::cout << ": size=" << qstore.size() << std::endl;
               size_t tsize_cpu = 0, tsize_gpu = 0;
               for(auto pr = qstore.cbegin(); pr != qstore.cend(); pr++){
                  bool avail_cpu = pr->second.avail_cpu();
                  bool avail_gpu = pr->second.avail_gpu();
                  std::cout << " fqop=" << pr->first 
                     << " cpu=" << avail_cpu
                     << " gpu=" << avail_gpu
                     << " size=" << pr->second.size()
                     << ":" << tools::sizeMB<Tm>(pr->second.size()) << "MB"
                     << ":" << tools::sizeGB<Tm>(pr->second.size()) << "GB"
                     << std::endl;
                  if(avail_cpu) tsize_cpu += pr->second.size();
                  if(avail_gpu) tsize_gpu += pr->second.size();
               }
               std::cout << " total size[cpu]=" << tsize_cpu
                  << ":" << tools::sizeMB<Tm>(tsize_cpu) << "MB"
                  << ":" << tools::sizeGB<Tm>(tsize_cpu) << "GB" 
                  << std::endl;
               std::cout << " total size[gpu]=" << tsize_gpu
                  << ":" << tools::sizeMB<Tm>(tsize_gpu) << "MB"
                  << ":" << tools::sizeGB<Tm>(tsize_gpu) << "GB"
                  << std::endl;
            }
            // fetch from disk to cpu/gpu memory
            void fetch_to_memory(const std::vector<std::string> fneed, const bool ifgpu);
            void fetch_to_cpumem(const std::vector<std::string> fneed_next, const bool async_fetch=false);
            // join and erase from cpu & gpu memory
            void join_and_erase(const std::vector<std::string> fneed, 
                  const std::vector<std::string> fneed_next={});
            // only clear cpu memory
            void clear_from_memory(const std::vector<std::string> fneed,
                  const std::vector<std::string> fneed_next); 
            void clear_from_cpumem(const std::vector<std::string> fneed,
                  const std::vector<std::string> fneed_next,
                  const bool ifkeepcoper); 
            // save renormalized operator to file 
            void save_to_disk(const std::string frop, const bool async_save, const bool async_tocpu, 
                  const std::vector<std::string> fneed_next={});
            // remove fdel [in the same bond as frop] from disk
            void remove_from_disk(const std::string fdel, const bool async_remove);
            // join
            void join_all(){
               if(thread_fetch.joinable()) thread_fetch.join();
               if(thread_save.joinable()) thread_save.join();
               if(thread_remove.joinable()) thread_remove.join();
            }
            // final call
            void finalize(){
               this->join_all();
               qstore.clear();
               frop_prev.clear();
            }
         private:
            int iomode=0;
            bool debug=false;
            operData_pool_raw<Tm> qstore;
            std::thread thread_fetch; // prefetch qops for the next dbond
            std::thread thread_save; // save renormalized operators
            std::thread thread_remove; // remove qops on the same bond with opposite direction
            std::string frop_prev;
      };

   template <typename Tm>
      void oper_pool_raw<Tm>::fetch_to_memory(const std::vector<std::string> fneed, const bool ifgpu){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::fetch_to_memory: ifgpu=" << ifgpu
               << " size=" << fneed.size() << std::endl;
            this->display("in");
         }
         for(int i=0; i<fneed.size(); i++){
            const auto& fqop = fneed[i];
            bool ifexist = this->exist(fqop);
            if(debug) std::cout << " fetch: i=" << i << " fqop=" << fqop << " ifexist=" << ifexist << std::endl;
            if(ifexist) continue;
            oper_load(iomode, fqop, qstore[fqop], debug);
         }
#ifdef GPU
         if(ifgpu){
            for(const auto& fqop : fneed){
               auto& tqops = qstore[fqop];
               if(tqops.avail_gpu()) continue;
               tqops.allocate_gpu();
               tqops.to_gpu();
            }
         }
#endif
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::fetch_to_memory : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_fetch(operData_pool_raw<Tm>& qstore,
            const std::vector<std::string> fneed,
            const std::vector<bool> fetch,
            const int iomode,
            const bool debug){
         // load new data is in memory
         for(int i=0; i<fneed.size(); i++){
            const auto& fqop = fneed[i];
            if(!fetch[i]) continue;
            if(debug) std::cout << " fetch: fqop=" << fqop << std::endl;
            oper_load(iomode, fqop, qstore[fqop], debug);
         }
      }

   template <typename Tm>
      void oper_pool_raw<Tm>::fetch_to_cpumem(const std::vector<std::string> fneed, const bool async_fetch){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::fetch_to_cpumem: async_fetch=" << async_fetch
               << " size=" << fneed.size() << std::endl;
            this->display("in");
         }
         std::vector<bool> fetch(fneed.size(),0); 
         for(int i=0; i<fneed.size(); i++){
            const auto& fqop = fneed[i];
            bool ifexist = this->exist(fqop);
            if(debug) std::cout << " fetch: i=" << i << " fqop=" << fqop << " ifexist=" << ifexist << std::endl;
            fetch[i] = !ifexist;
            qstore[fqop]; // declare a spot here! this is helpful for threadsafty
         }
         assert(!thread_fetch.joinable());
         if(!async_fetch){
            oper_fetch(qstore, fneed, fetch, iomode, debug);
         }else{
            thread_fetch = std::thread(&ctns::oper_fetch<Tm>, std::ref(qstore), fneed, fetch, iomode, debug);
         }
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::fetch_to_cpumem : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   // release unnecessary qops in the next point
   template <typename Tm>
      void oper_pool_raw<Tm>::join_and_erase(const std::vector<std::string> frelease,
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::join_and_erase : size=" << frelease.size() << std::endl; 
            for(const auto& fqop : frelease){
               bool ifexist = this->exist(fqop);
               auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop); 
               bool iferase = (result == fneed_next.end()) && (fqop != frop_prev); 
               std::cout << " erase: fqop=" << fqop << " ifexist=" << ifexist << " iferase=" << iferase << std::endl;
            }
            this->display("in");
         }
         this->join_all();
         auto t1 = tools::get_time();
         for(const auto& fqop : frelease){
            if(fqop == frop_prev) continue;
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            if(result != fneed_next.end()) continue;
            qstore.erase(fqop);
         }
         auto t2 = tools::get_time();
         // if result is not used in the next dbond, then release it
         // NOTE: check is neceesary at the returning point: [ -*=>=*-* and -*=<=*-* ],
         // because the previous left qops is needed in the next dbond!
         auto result = std::find(fneed_next.begin(), fneed_next.end(), frop_prev);
         if(result == fneed_next.end()){
            qstore.erase(frop_prev); // NOTE: frop_prev is only erased here to make sure the saving is finished!
         }
         if(debug){
            this->display("out");
            auto t3 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::join_and_erase : "
               << tools::get_duration(t3-t0) << " S"
               << " T(join/erase1/erase2)="
               << tools::get_duration(t1-t0) << ","
               << tools::get_duration(t2-t1) << ","
               << tools::get_duration(t3-t2)
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool_raw<Tm>::clear_from_memory(const std::vector<std::string> fclear,
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::clear_from_memory: size=" << fclear.size() << std::endl; 
            for(const auto& fqop : fclear){
               bool ifexist = this->exist(fqop);
               auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop); 
               bool ifclear = (result == fneed_next.end()) && (fqop != frop_prev); 
               std::cout << " clear: fqop=" << fqop << " ifexist=" << ifexist << " ifclear=" << ifclear << std::endl;
            }
            this->display("in");
         }
         for(const auto& fqop : fclear){
            if(fqop == frop_prev) continue; // DO NOT remove CPU space, since saving may not finish!
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            if(result != fneed_next.end()) continue;
            qstore[fqop].clear();
            qstore[fqop].clear_gpu();
            qstore[fqop]._size = 0;
         }
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::clear_from_memory : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool_raw<Tm>::clear_from_cpumem(const std::vector<std::string> fclear,
            const std::vector<std::string> fneed_next,
            const bool ifkeepcoper){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::clear_from_cpumem: size=" << fclear.size() << std::endl; 
            for(const auto& fqop : fclear){
               bool ifexist = this->exist(fqop);
               auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop); 
               bool ifclear = (result == fneed_next.end()) && (fqop != frop_prev); 
               std::cout << " clear: fqop=" << fqop << " ifexist=" << ifexist << " ifclear=" << ifclear << std::endl;
            }
            this->display("in");
         }
         for(int i=0; i<fclear.size(); i++){
            if(ifkeepcoper && i >= 2) continue; // skip op[c2/c1]
            const auto& fqop = fclear[i];
            if(fqop == frop_prev) continue; // DO NOT remove CPU space, since saving may not finish!
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            if(result != fneed_next.end()) continue;
            qstore[fqop].clear();
         }
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::clear_from_cpumem : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_dump(oper_dict<Tm>& qops,
            const std::string frop,
            const bool async_tocpu,
            const int iomode,
            const bool debug){
#ifdef GPU
         if(async_tocpu){
            qops.allocate_cpu();
            qops.to_cpu();
         }
#endif
         oper_save<Tm>(iomode, frop, qops, debug);
      }

   // save to disk
   template <typename Tm>
      void oper_pool_raw<Tm>::save_to_disk(const std::string frop, const bool async_save, 
            const bool async_tocpu,
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::save_to_disk: async_save=" << async_save 
               << " async_tocpu=" << async_tocpu << " frop=" << frop << std::endl;
         }
         assert(!thread_save.joinable()); 
         if(!async_save){
            oper_dump<Tm>(qstore[frop], frop, async_tocpu, iomode, debug);
         }else{
            thread_save = std::thread(&ctns::oper_dump<Tm>, std::ref(qstore[frop]), frop, async_tocpu, iomode, debug);
         }
         frop_prev = frop;
         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::save_to_disk : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool_raw<Tm>::remove_from_disk(const std::string fdel, const bool async_remove){
         if(debug){
            std::cout << "ctns::oper_pool_raw<Tm>::remove_from_disk: async_remove=" << async_remove 
               << " fdel=" << fdel << std::endl; 
         }
         auto t0 = tools::get_time();
         assert(!thread_remove.joinable()); 
         if(!async_remove){
            ctns::oper_remove(fdel, debug);
         }else{
            thread_remove = std::thread(&ctns::oper_remove, fdel, debug);
         }
         if(debug){
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool_raw<Tm>::remove_from_disk : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

} // ctns

#endif
