#ifndef OPER_POOL_H
#define OPER_POOL_H

#include <thread>
#include "oper_dict.h"
#include "oper_io.h"

namespace ctns{
    
   // pool for mananging operators
   template <typename Tm>
      struct oper_pool{
         public:
            // constuctor
            oper_pool(const int _iomode, const bool _debug): iomode(_iomode), debug(_debug) {}
            const oper_dict<Tm>& operator()(const std::string fqop) const{
               return qstore.at(fqop);
            } 
            oper_dict<Tm>& operator()(const std::string fqop){
               return qstore[fqop];
            }
            // total size allocated for storing operators
            size_t size() const{
               size_t sz = 0;
               for(const auto& pr : qstore){
                  sz += pr.second.size();
               }
               return sz;
            }
            void display(const std::string msg) const{
               std::cout << "qstore";
               if(msg.size()>0) std::cout << "[" << msg << "]";
               std::cout << ": size=" << qstore.size() << std::endl;
               size_t tsize = 0; 
               for(const auto& pr : qstore){
                  std::cout << " fqop=" << pr.first 
                     << " size=" << pr.second.size()
                     << ":" << tools::sizeMB<Tm>(pr.second.size()) << "MB"
                     << ":" << tools::sizeGB<Tm>(pr.second.size()) << "GB"
                     << " cpu=" << pr.second.avail_cpu()
                     << " gpu=" << pr.second.avail_gpu()
                     << std::endl;
                  tsize += pr.second.size();
               }
               std::cout << " total size=" << tsize
                  << ":" << tools::sizeMB<Tm>(tsize) << "MB"
                  << ":" << tools::sizeGB<Tm>(tsize) << "GB"
                  << std::endl;
            }
            // fetch qops from memory / disk
            void fetch(const std::vector<std::string> fneed, const bool ifgpu, const bool ifasync=false);
            // release
            void release(const std::vector<std::string> frelease);
            // release
            void release(const std::vector<std::string> fneed, 
                  const std::vector<std::string> fneed_next);
            // save to disk
            void save(const std::string frop, const bool ifasync);
            // remove from disk
            void remove(const std::string fdel, const bool ifasync);
            // finalize
            void finalize(){
               if(thread_fetch.joinable()) thread_fetch.join();
               if(thread_save.joinable()) thread_save.join();
               if(thread_remove.joinable()) thread_remove.join();
               qstore.clear();
            }
         public:
            int iomode=0;
            bool debug=false;
            std::map<std::string,oper_dict<Tm>> qstore;
            std::thread thread_fetch; // prefetch qops for the next dbond
            std::thread thread_save; // save renormalized operators
            std::thread thread_remove; // remove qops on the same bond with opposite direction
            std::string frop_prev;
      };

   template <typename Tm>
      void oper_fetch(std::map<std::string,oper_dict<Tm>>& qstore,
                 const std::vector<std::string> fneed, 
                 const bool ifgpu,
                 const int iomode,
                 const bool debug){
         // load new data is in memory
         for(const auto& fqop : fneed){
            // IMPORTANT: do not load if fqop is already in memory
            if(qstore.find(fqop) != qstore.end()) continue;
            if(debug) std::cout << "load: fqop=" << fqop << std::endl;
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
      }

   template <typename Tm>
      void oper_pool<Tm>::fetch(const std::vector<std::string> fneed, const bool ifgpu, const bool ifasync){
         auto t0 = tools::get_time();
         if(debug){
            std::cout << "ctns::oper_pool<Tm>::fetch: ifgpu=" << ifgpu << " ifasyn=" << ifasync
               << " fneed size=" << fneed.size() << std::endl;
            for(const auto& fqop : fneed){
               bool ifexist = qstore.find(fqop) != qstore.end();
               std::cout << " fqop=" << fqop << " ifexist=" << ifexist << std::endl;
            }
            this->display("in");
         }
         if(thread_fetch.joinable()) thread_fetch.join();
         auto t1 = tools::get_time();
         if(!ifasync){
            oper_fetch(qstore, fneed, ifgpu, iomode, debug);
         }else{
            thread_fetch = std::thread(&ctns::oper_fetch<Tm>, std::ref(qstore), fneed, ifgpu, iomode, debug);
         }
         if(debug){
            this->display("out");
            auto t2 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool<Tm>::fetch : "
               << tools::get_duration(t2-t0) << " S"
               << " T(sync/fetch)="
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << " -----"
               << std::endl;
         }
      }

   template <typename Tm>
      void oper_pool<Tm>::release(const std::vector<std::string> frelease){
         if(debug){
            std::cout << "ctns::oper_pool<Tm>::release size=" << frelease.size() << std::endl; 
            for(const auto& fqop : frelease){
               bool ifexist = qstore.find(fqop) != qstore.end();
               std::cout << " fqop=" << fqop << " ifexist=" << ifexist << std::endl;
            }
            this->display("in");
         }
         for(const auto& fqop : frelease){
            qstore.erase(fqop);
         }
         if(debug) this->display("out");
      }

   // release unnecessary qops in the next point
   template <typename Tm>
      void oper_pool<Tm>::release(const std::vector<std::string> fneed,
            const std::vector<std::string> fneed_next){
         std::vector<std::string> frelease;
         for(const auto& fqop : fneed){
            auto result = std::find(fneed_next.begin(), fneed_next.end(), fqop);
            // if result is not used in the next dbond, then release it
            if(result == fneed_next.end()){
               frelease.push_back(fqop);
            }
         }
         this->release(frelease);
      }

   // save to disk
   template <typename Tm>
      void oper_pool<Tm>::save(const std::string frop, const bool ifasync){
         if(debug){
            std::cout << "ctns::oper_pool<Tm>::save ifasync=" << ifasync << " frop=" << frop 
               << " erase frop_prev=" << frop_prev << std::endl;
         }
         auto t0 = tools::get_time();
         if(thread_save.joinable()) thread_save.join(); // join before erasing the last rop! 
         auto t1 = tools::get_time();
         if(!ifasync){
            ctns::oper_save<Tm>(iomode, frop, qstore.at(frop), debug);
         }else{
            thread_save = std::thread(&ctns::oper_save<Tm>, iomode, frop,
                  std::cref(qstore.at(frop)), debug);
         }
         auto t2 = tools::get_time();
         qstore.erase(frop_prev);
         frop_prev = frop;
         if(debug){
            auto t3 = tools::get_time();
            std::cout << "----- TIMING FOR oper_pool<Tm>::save : "
               << tools::get_duration(t3-t0) << " S"
               << " T(sync/save/erase)=" 
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << "," 
               << tools::get_duration(t3-t2) << " -----"
               << std::endl;
         }
      }

   // remove from disk
   template <typename Tm>
      void oper_pool<Tm>::remove(const std::string fdel, const bool ifasync){
         if(debug){
            std::cout << "ctns::oper_pool<Tm>::remove ifasync=" << ifasync << " fdel=" << fdel << std::endl; 
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
            std::cout << "----- TIMING FOR oper_pool<Tm>::remove : "
               << tools::get_duration(t2-t0) << " S"
               << " T(sync/remove)=" 
               << tools::get_duration(t1-t0) << "," 
               << tools::get_duration(t2-t1) << " -----"
               << std::endl;
         }
      }

} // ctns

#endif
