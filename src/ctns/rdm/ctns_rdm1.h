#ifndef CTNS_RDM1_H
#define CTNS_RDM1_H

#include "rdm_env.h"
#include "rdm_util.h"
#include "../sweep_init.h"
#ifndef SERIAL
#include "../core/mpi_wrapper.h"
#endif

namespace ctns{

   template <typename Qm, typename Tm>
      void get_rdm1(const bool is_same,
            const comb<Qm,Tm>& icomb,
            const comb<Qm,Tm>& icomb2,
            const input::schedule& schd,
            const std::string scratch,
            linalg::matrix<Tm>& rdm1){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif
         const bool ifab = Qm::ifabelian;
         const bool debug = (rank==0); 
         if(debug){ 
            std::cout << "\nctns::get_rdm1 ifab=" << ifab
               << std::endl;
         }
         auto t0 = tools::get_time();


         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::get_rdm1", t0, t1);
         }
      }

} // ctns

#endif
