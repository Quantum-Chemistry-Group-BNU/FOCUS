#ifndef SWEEP_RESTART_H
#define SWEEP_RESTART_H

#include "../core/tools.h"
#include "../core/linalg.h"
#include "qtensor/qtensor.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   // onedot optimization algorithm
   template <typename Km>
      void sweep_restart(comb<Km>& icomb,
            const integral::two_body<typename Km::dtype>& int2e,
            const integral::one_body<typename Km::dtype>& int1e,
            const double ecore,
            const input::schedule& schd,
            const std::string scratch,
            oper_pool<typename Km::dtype>& qops_pool,
            sweep_data& sweeps,
            const int isweep,
            const int ibond){
         using Tm = typename Km::dtype;
         int rank = 0, size = 1, maxthreads = 1;
#ifndef SERIAL
         rank = icomb.world.rank();
         size = icomb.world.size();
#endif   
#ifdef _OPENMP
         maxthreads = omp_get_max_threads();
#endif
         const bool debug = (rank==0);
         if(debug){
            std::cout << "ctns::sweep_restart"
               << " mpisize=" << size
               << " maxthreads=" << maxthreads 
               << std::endl;
         }
         auto t0 = tools::get_time();

         //        
         // Because some renormalized operators are deleted during the sweep, we need
         // to rebuild the renormalized operators along the sweep via renormalization.
         //

         // 0. check partition 
         const auto& dbond = sweeps.seq[ibond];
         icomb.topo.check_partition(1, dbond, debug, schd.ctns.verbose);

         // 1. load site (only for rank 0)
         const auto p = dbond.get_current();
         const auto& pdx = icomb.topo.rindex.at(p);
         if(rank == 0) sweep_load(icomb, schd, scratch, sweeps, isweep, ibond);
#ifndef SERIAL
         if(size > 1) mpi_wrapper::broadcast(icomb.world, icomb.sites[pdx], 0);
#endif

         // 2. load operators & renorm
         auto fneed = icomb.topo.get_fqops(1, dbond, scratch, false); // lrc
         auto frop = icomb.topo.get_fbond(dbond, scratch, false).first;
         std::string superblock, fname;
         if(dbond.forward){
            superblock = dbond.is_cturn()? "lr" : "lc";
            if(superblock == "lc") fneed[1] = fneed[2];
         }else{
            superblock = "cr";
            fneed[0] = fneed[2];
         }
         fneed.resize(2);
         qops_pool.fetch_to_memory(fneed, schd.ctns.alg_renorm>10);
         
         dot_timing timing_local;
         oper_renorm(superblock, icomb, p, int2e, int1e, schd,
               qops_pool.at(fneed[0]), qops_pool.at(fneed[1]), qops_pool[frop], 
               fname, timing_local); 
         
         qops_pool.join_and_erase(fneed);
         qops_pool.save_to_disk(frop, schd.ctns.alg_renorm>10 && schd.ctns.async_tocpu, schd.ctns.async_save);

         auto t1 = tools::get_time();
         if(debug) tools::timing("ctns::sweep_restart", t0, t1);
      }

} // ctns

#endif
