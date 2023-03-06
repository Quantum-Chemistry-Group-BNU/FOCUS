#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../core/serialization.h"
#include "ctns_comb.h"

namespace ctns{ 

   const bool ifsavebin = false;
   extern const bool ifsavebin;

   // for comb
   template <typename Km>
      void rcanon_save(const comb<Km>& icomb,
            const std::string fname="rcanon.info"){
         std::cout << "\nctns::rcanon_save fname=" << fname << std::endl;

         std::ofstream ofs(fname, std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         // save sites 
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            save << icomb.sites[idx];
         }
         ofs.close();
         
         // ZL@20221207 binary format for easier loading in python 
         if(ifsavebin){
            std::ofstream ofs2(fname+".bin", std::ios::binary);
            ofs2.write((char*)(&icomb.topo.ntotal), sizeof(int));
            // save all sites
            for(int idx=0; idx<icomb.topo.ntotal; idx++){
               icomb.sites[idx].dump(ofs2);
            }
            ofs2.close();
         }
      }

   template <typename Km>
      void rcanon_load(comb<Km>& icomb, // no const!
            const std::string fname="rcanon.info"){
         std::cout << "\nctns:rcanon_load fname=" << fname << std::endl;

         std::ifstream ifs(fname, std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         // load sites 
         icomb.sites.resize(icomb.topo.ntotal);
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            load >> icomb.sites[idx]; // this save calls to copy constructor for vector<st3>
         }
         ifs.close();
      }

} // ctns

#endif
