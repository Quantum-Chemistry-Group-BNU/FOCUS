#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../core/serialization.h"
#include "ctns_comb.h"

namespace ctns{ 

   const bool ifsavebin = false;
   extern const bool ifsavebin;

   // for comb
   template <typename Qm, typename Tm>
      void rcanon_save(const comb<Qm,Tm>& icomb,
            const std::string fname="rcanon.info"){
         std::cout << "\nctns::rcanon_save fname=" << fname << std::endl;

         std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         // save sites 
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            save << icomb.sites[idx];
         }
         save << icomb.rwfuns;
         ofs.close();
         
         // ZL@20221207 binary format for easier loading in python 
         if(ifsavebin){
            std::ofstream ofs2(fname+".bin", std::ios::binary);
            ofs2.write((char*)(&icomb.topo.ntotal), sizeof(int));
            // save all sites
            for(int idx=0; idx<icomb.topo.ntotal-1; idx++){
               icomb.sites[idx].dump(ofs2);
            }
            // merge rwfun & site0
            const auto& rindex = icomb.topo.rindex;
            const auto& site0 = icomb.sites[rindex.at(std::make_pair(0,0))];
            auto site = contract_qt3_qt2("l",site0,icomb.get_wf2());
            site.dump(ofs2);
            ofs2.close();
         }
      }

   template <typename Qm, typename Tm>
      void rcanon_load(comb<Qm,Tm>& icomb, // no const!
            const std::string fname="rcanon.info"){
         std::cout << "\nctns:rcanon_load fname=" << fname << std::endl;

         std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         // load sites 
         icomb.sites.resize(icomb.topo.ntotal);
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            load >> icomb.sites[idx]; // this save calls to copy constructor for vector<st3>
         }
         load >> icomb.rwfuns;
         ifs.close();
      }

   // for site & cpsi
   template <typename Tm>
      void rcanon_save(const Tm& site,
            const std::string fname){
         std::cout << "ctns::rcanon_save fname=" << fname << std::endl;
         std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         save << site;
         ofs.close();
      }

   template <typename Tm>
      void rcanon_load(Tm& site,
            const std::string fname){
         std::cout << "ctns:rcanon_load fname=" << fname << std::endl;
         std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         load >> site;
         ifs.close();
      }

} // ctns

#endif
