#ifndef CTNS_IO_H
#define CTNS_IO_H

#include "../io/input.h"
#include "../io/io.h"
#include "../core/serialization.h"
#include "ctns_comb.h"

namespace ctns{ 

   // for comb
   template <typename Qm, typename Tm>
      void rcanon_save(const comb<Qm,Tm>& icomb,
            const std::string fname,
            const bool debug=true){
         if(debug){
            std::cout << "\nctns::rcanon_save fname=" << fname;
         }
         auto t0 = tools::get_time();
         std::ofstream ofs(fname+".info", std::ios::binary);
         boost::archive::binary_oarchive save(ofs);
         // save sites 
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            save << icomb.sites[idx];
         }
         save << icomb.rwfuns;
         ofs.close();
         if(debug){
            auto t1 = tools::get_time();
            double dt = tools::get_duration(t1-t0);
            std::cout << std::setprecision(2) << " T(save)=" << dt << "S" << std::endl; 
         }
      }

   // ZL@20221207 binary format for easier loading in python 
   template <typename Qm, typename Tm>
      void rcanon_savebin(const comb<Qm,Tm>& icomb,
            const std::string fname,
            const bool debug=true){
         if(debug) std::cout << "\nctns::rcanon_savebin fname=" << fname << std::endl;
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

   template <typename Qm, typename Tm>
      void rcanon_load(comb<Qm,Tm>& icomb, // no const!
            const std::string fname,
            const bool debug=true){
         if(debug){
            std::cout << "\nctns:rcanon_load fname=" << fname;
         }
         auto t0 = tools::get_time();

         io::file_existQ(fname+".info"); // check existance first

         std::ifstream ifs(fname+".info", std::ios::binary);
         boost::archive::binary_iarchive load(ifs);
         // load sites 
         icomb.sites.resize(icomb.topo.ntotal);
         for(int idx=0; idx<icomb.topo.ntotal; idx++){
            load >> icomb.sites[idx]; // this save calls to copy constructor for vector<st3>
         }
         load >> icomb.rwfuns;
         ifs.close();

         if(debug){
            auto t1 = tools::get_time();
            double dt = tools::get_duration(t1-t0);
            std::cout << std::setprecision(2) << " T(load)=" << dt << "S" << std::endl;
         } 
      }

   // load CTNS
   template <typename Qm, typename Tm>
      void comb_load(comb<Qm,Tm>& icomb, // no const!
            const input::schedule& schd,
            const std::string fname){
         int rank = 0, size = 1;
#ifndef SERIAL
         rank = schd.world.rank();
         size = schd.world.size();
#endif
         const bool debug = (rank==0) && schd.ctns.verbose>0; 
         if(rank == 0){
            assert(fname.size() > 0);
            auto rcanon_file = schd.scratch+"/"+fname;
            std::cout << "\nctns::comb_load fname=" << fname << std::endl;
           
            // load topology
            if(!schd.ctns.topology_file.empty()){
               icomb.topo.read(schd.ctns.topology_file, debug);
            }else{
               icomb.topo.gen1d(schd.sorb/2);
            }
            if(debug) icomb.topo.print();
            
            // load RCF
            rcanon_load(icomb, rcanon_file, debug);
            if(debug) rcanon_check(icomb, schd.ctns.thresh_ortho);
         } // rank 0
#ifndef SERIAL
         if(size > 1){
            mpi_wrapper::broadcast(schd.world, icomb, 0);
            icomb.world = schd.world;
         }
#endif
      }

} // ctns

#endif
