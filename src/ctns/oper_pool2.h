#ifndef OPER_POOL2_H
#define OPER_POOL2_H

#include <thread>
#include <memory>
#include "oper_dict.h"
#include "oper_io.h"

#include "safe_ptr.h"

namespace ctns{

   template <typename Tm>
      using operData_pool2 = sf::safe_ptr< std::map<std::string,oper_dict<Tm>> >;

   // pool for mananging operators
   template <typename Tm>
      struct oper_pool2{
         public:
            // constuctor
            oper_pool2(const int _iomode, const bool _debug): iomode(_iomode), debug(_debug){}
            const oper_dict<Tm>& at(const std::string fqop) const{
               assert(this->exist(fqop));
               return qstore->at(fqop);
            }
            oper_dict<Tm>& operator()(const std::string& fqop){
               return (*qstore)[fqop];
            }
            bool exist(const std::string& frop) const{
               return qstore->find(frop) != qstore->end();
            }
            // total size allocated for storing operators
            size_t size() const{
               size_t sz = 0;
               for(auto pr = qstore->cbegin(); pr != qstore->cend(); pr++){
                  sz += pr->second.size();
               }
               return sz;
            }
            void display(const std::string msg="") const{
               std::cout << "qstore";
               if(msg.size()>0) std::cout << "[" << msg << "]";
               std::cout << ": size=" << qstore->size() << std::endl;
               size_t tsize_cpu = 0, tsize_gpu = 0;
               for(auto pr = qstore->cbegin(); pr != qstore->cend(); pr++){
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
            void fetch_to_memory(const std::vector<std::string> fneed, const bool ifgpu, const bool ifasync=false);
            void erase_from_memory(const std::vector<std::string> fneed, 
                  const std::vector<std::string> fneed_next={});
            void clear_from_cpumem(const std::vector<std::string> fneed,
                  const std::vector<std::string> fneed_next={}); 
            void save_to_disk(const std::string frop, const bool ifgpu, const bool ifasync, 
                  const std::vector<std::string> fneed_next={});
            // remove fdel [in the same bond as frop] from disk
            void remove_from_disk(const std::string fdel, const bool ifasync);
            void finalize(){
               if(thread_fetch.joinable()) thread_fetch.join();
               if(thread_save.joinable()) thread_save.join();
               if(thread_remove.joinable()) thread_remove.join();
               qstore->clear();
            }
         public:
            int iomode=0;
            bool debug=false;
            operData_pool2<Tm> qstore;
            std::thread thread_fetch; // prefetch qops for the next dbond
            std::thread thread_save; // save renormalized operators
            std::thread thread_remove; // remove qops on the same bond with opposite direction
            std::string frop_prev;
      };

   template <typename Tm>
      void oper_fetch(operData_pool2<Tm>& qstore,
            const std::vector<std::string> fneed,
            const std::vector<bool> fetch,
            const bool ifgpu,
            const int iomode,
            const bool debug){
         // load new data is in memory
         for(int i=0; i<fneed.size(); i++){
            const auto& fqop = fneed[i];
            if(!fetch[i]) continue;
            if(debug) std::cout << "load: fqop=" << fqop << std::endl;
            oper_load(iomode, fqop, (*qstore)[fqop], debug);
         }
#ifdef GPU
         if(ifgpu){
            for(const auto& fqop : fneed){
               auto& tqops = (*qstore)[fqop];
               if(tqops.avail_gpu()) continue;
               tqops.allocate_gpu();
               tqops.to_gpu();
            }
         }
#endif
      }

   template <typename Tm>
      void oper_pool2<Tm>::fetch_to_memory(const std::vector<std::string> fneed, const bool ifgpu, const bool ifasync){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool2<Tm>::fetch_to_memory: ifgpu=" << ifgpu << " ifasyn=" << ifasync
               << " fneed size=" << fneed.size() << std::endl;
            this->display("in");
         }
         std::vector<bool> fetch(fneed.size(),0); 
         for(int i=0; i<fneed.size(); i++){
            const auto& fqop = fneed[i];
            bool ifexist = this->exist(fqop);
            if(debug) std::cout << " i=" << i << " fqop=" << fqop << " ifexist=" << ifexist << std::endl;
            fetch[i] = !ifexist;
            (*qstore)[fqop]; // declare a spot here! this is helpful for threadsafty
         }
         if(thread_fetch.joinable()) thread_fetch.join();
         auto t1 = tools::get_time();
         if(!ifasync){
            oper_fetch(qstore, fneed, fetch, ifgpu, iomode, debug);
         }else{
            thread_fetch = std::thread(&ctns::oper_fetch<Tm>, std::ref(qstore), fneed, fetch, ifgpu, iomode, debug);
         }
         if(debug){
            this->display("out");
            auto t2 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool2<Tm>::fetch_to_memory : "
               << tools::get_duration(t2-t0) << " S"
               << " T(sync/fetch)="
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << " -----"
               << std::endl;
         }
      }

   // release unnecessary qops in the next point
   template <typename Tm>
      void oper_pool2<Tm>::erase_from_memory(const std::vector<std::string> frelease,
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool2<Tm>::erase_from_memory: size=" << frelease.size() << std::endl; 
            for(const auto& fqop : frelease){
               bool ifexist = this->exist(fqop);
               auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop); 
               bool iferase = (result == fneed_next.end()) && (fqop != frop_prev); 
               std::cout << " fqop=" << fqop << " ifexist=" << ifexist << " iferase=" << iferase << std::endl;
            }
            this->display("in");
         }
         for(const auto& fqop : frelease){
            if(fqop == frop_prev) continue; // DO NOT remove CPU space, since saving may not finish!
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            if(result != fneed_next.end()) continue;
            qstore->erase(fqop);
         }
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool2<Tm>::erase_from_memory : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool2<Tm>::clear_from_cpumem(const std::vector<std::string> fclear,
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool2<Tm>::clear_from_cpumem: size=" << fclear.size() << std::endl; 
            for(const auto& fqop : fclear){
               bool ifexist = this->exist(fqop);
               auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop); 
               bool ifclear = (result == fneed_next.end()) && (fqop != frop_prev); 
               std::cout << " fqop=" << fqop << " ifexist=" << ifexist << " ifclear=" << ifclear << std::endl;
            }
            this->display("in");
         }
         for(auto& fqop : fclear){
            if(fqop == frop_prev) continue; // DO NOT remove CPU space, since saving may not finish!
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            if(result != fneed_next.end()) continue;
            (*qstore)[fqop].clear();
         }
         if(debug){
            this->display("out");
            auto t1 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool2<Tm>::clear_from_cpumem : "
               << tools::get_duration(t1-t0) << " S -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_dump(operData_pool2<Tm>& qstore,
            const std::string frop,
            const bool ifgpu,
            const int iomode,
            const bool debug){
#ifdef GPU
         if(ifgpu) (*qstore)[frop].to_cpu();
#endif
         oper_save<Tm>(iomode, frop, qstore->at(frop), debug);
      }

   // save to disk
   template <typename Tm>
      void oper_pool2<Tm>::save_to_disk(const std::string frop, const bool ifgpu, const bool ifasync, 
            const std::vector<std::string> fneed_next){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool2<Tm>::save_to_disk: ifgpu=" << ifgpu << " ifasync=" << ifasync 
               << " frop=" << frop << " erase frop_prev=" << frop_prev << std::endl;
         }
         if(thread_save.joinable()) thread_save.join(); // join before erasing the last rop! 
         auto t1 = tools::get_time();
         if(!ifasync){
            oper_dump<Tm>(qstore, frop, ifgpu, iomode, debug);
         }else{
            thread_save = std::thread(&ctns::oper_dump<Tm>, std::ref(qstore), frop, ifgpu, iomode, debug);
         }
         auto t2 = tools::get_time();
         // if result is not used in the next dbond, then release it
         // NOTE: check is neceesary at the returning point: [ -*=>=*-* and -*=<=*-* ],
         // because the previous left qops is needed in the next dbond!
         auto result = std::find(fneed_next.begin(), fneed_next.end(), frop_prev);
         if(result == fneed_next.end()){
            qstore->erase(frop_prev); // NOTE: frop_prev is only erased here to make sure the saving is finished!!!
         }
         frop_prev = frop;
         if(debug){
            auto t3 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool2<Tm>::save_to_disk : "
               << tools::get_duration(t3-t0) << " S"
               << " T(sync/save/erase)=" 
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << "," 
               << tools::get_duration(t3-t2) << " -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool2<Tm>::remove_from_disk(const std::string fdel, const bool ifasync){
         if(debug){
            std::cout << "ctns::oper_pool2<Tm>::remove_from_disk ifasync=" << ifasync << " fdel=" << fdel << std::endl; 
         }
         auto t0 = tools::get_time();
         if(thread_remove.joinable()) thread_remove.join();
         auto t1 = tools::get_time();
         if(!ifasync){
            ctns::oper_remove(fdel, debug);
         }else{
            thread_remove = std::thread(&ctns::oper_remove, fdel, debug);
         }
         if(debug){
            auto t2 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool2<Tm>::remove_from_disk : "
               << tools::get_duration(t2-t0) << " S"
               << " T(sync/remove)=" 
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
