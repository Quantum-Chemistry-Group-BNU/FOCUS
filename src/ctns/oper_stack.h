#ifndef OPER_STACK_H
#define OPER_STACK_H

#include <thread>
#include "oper_dict.h"
#include "oper_io.h"

namespace ctns{

template <typename Tm>
struct oper_stack{
   public:
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
      void fetch(const std::vector<std::string>& fneed,
		 const bool debug);
      void save(const std::string frop,
		const bool debug){
         if(thrd.joinable()) thrd.join();
	 fkept = frop;

	 thrd = std::thread(&ctns::oper_save<Tm>, frop,
			    std::cref(qstore.at(frop)), debug);
/*
	 ctns::oper_save<Tm>(frop, qstore.at(frop), debug);
*/
	 // we can also add HDF5 support here

      }
      void clean_up(){
	 // must be first join then clear, otherwise IO is not finished!
	 if(thrd.joinable()) thrd.join();
	 qstore.clear();
      }
   public:
      std::map<std::string,oper_dict<Tm>> qstore; // for global storage
      std::string fkept;
      std::thread thrd;
};

template <typename Tm>
void oper_stack<Tm>::fetch(const std::vector<std::string>& fneed,
		 	   const bool debug){
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
   for(const auto& fqop : fneed){
      if(qstore.find(fqop) != qstore.end()) continue;
      oper_load(fqop, qstore[fqop], debug);
   }
   if(debug_oper_io && debug){
      this->display("out");
   }
}

} // ctns

#endif
