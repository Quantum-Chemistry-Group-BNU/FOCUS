#ifndef OPER_STACK_H
#define OPER_STACK_H

#include <thread>
#include "oper_dict.h"
#include "oper_io.h"

namespace ctns{

template <typename Tm>
struct oper_stack{
   public:
      // constuctor
      oper_stack(const int _iomode, const bool _debug): iomode(_iomode), debug(_debug) {}
      const oper_dict<Tm>& operator()(const std::string fqop) const{
         return qstore.at(fqop);
      } 
      oper_dict<Tm>& operator()(const std::string fqop){
	 return qstore[fqop];
      }
      void display(const std::string msg) const{
         std::cout << "qstore";
	 if(msg.size()>0) std::cout << "[" << msg << "]";
         std::cout << ": size=" << qstore.size() << std::endl; 
         for(const auto& pr : qstore){
            std::cout << " fqop=" << pr.first << std::endl;
         }
      }
      // fetch qops from memory / disk
      void fetch(const std::vector<std::string>& fneed);
      void save(const std::string frop){
	 fkept = frop;
         // just add if to be compatible with serial version
	 if(thrd.joinable()) thrd.join();
 	 ctns::oper_save<Tm>(iomode, fkept, qstore.at(fkept), debug);
/*
         thrd = std::thread(&ctns::oper_save<Tm>, iomode, fkept, 
			 std::cref(qstore.at(fkept)), debug);
*/

      }
      void clean_up(){
         // just add if to be compatible with serial version
	 if(thrd.joinable()) thrd.join();
/*
	 // dump the last file
	 ctns::oper_save<Tm>(iomode, fkept, qstore.at(fkept), debug);
*/
	 fkept.clear();
	 // must be first join then clear, otherwise IO is not finished!
	 qstore.clear();
      }
   public:
      int iomode;
      bool debug = false;
      std::map<std::string,oper_dict<Tm>> qstore; // for global storage
      std::string fkept;
      std::thread thrd;
};

template <typename Tm>
void oper_stack<Tm>::fetch(const std::vector<std::string>& fneed){
   auto t0 = tools::get_time();
   if(debug_oper_io && debug){
      std::cout << "ctns::oper_stack<Tm>::fetch" << std::endl;
      std::cout << "fneed: size=" << fneed.size() << std::endl;
      for(const auto& fqop : fneed){
	 std::cout << " fqop=" << fqop << std::endl;
      }
      this->display("in");
   }
   // first release uncessary qops
   std::vector<std::string> frelease;
   for(auto& pr : qstore){
      auto& fqop = pr.first;
      if(fqop == fkept) continue; // keep fkept in memory until IO is finished
      auto result = std::find(fneed.begin(), fneed.end(), fqop);
      if(result == fneed.end()){
         frelease.push_back(fqop);
      }
   }
   // just join before release   
   auto ta = tools::get_time();

//   if(thrd.joinable()) thrd.join();

   auto tb = tools::get_time();
   for(const auto& fqop : frelease){
      qstore.erase(fqop);
   }
   if(debug_oper_io && debug){
      std::cout << "frelease: size=" << frelease.size() << std::endl; 
      for(const auto& fqop : frelease){
         std::cout << " fqop=" << fqop << std::endl;
      }
   }
   // then load new data is in memory
   auto tc = tools::get_time();
   for(const auto& fqop : fneed){
      if(qstore.find(fqop) != qstore.end()) continue;
      oper_load(iomode, fqop, qstore[fqop], debug);
   }
   auto td = tools::get_time();
   if(debug_oper_io && debug) this->display("out");
/*
   // save the previous renormalized operators
   if(fkept.size() > 0){
      thrd = std::thread(&ctns::oper_save<Tm>, iomode, fkept, 
			 std::cref(qstore.at(fkept)), debug);
   }
*/
   auto t1 = tools::get_time();
   std::cout << "T(sync/load/tot)="
	     << tools::get_duration(tb-ta) << ","
	     << tools::get_duration(td-tc) << ","
	     << tools::get_duration(t1-t0) 
	     << std::endl;
   if(debug) tools::timing("ctns::oper_stack<Tm>::fetch", t0, t1);
}

} // ctns

#endif
