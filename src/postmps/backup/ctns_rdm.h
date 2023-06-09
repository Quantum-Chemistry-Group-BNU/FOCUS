#ifndef CTNS_RDM_H
#define CTNS_RDM_H

namespace ctns{

   template <typename Km>
      void sweep_rdm(const comb<Km>& icomb, // initial comb wavefunction
            const input::schedule& schd,
            const std::string scratch){
         using Tm = typename Km::dtype;
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const bool debug = (rank==0); 
         if(debug) std::cout << "\nctns::sweep_rdm" << std::endl;
         auto t0 = tools::get_time();

         // compute reduce density matrix

         exit(1);

         if(debug){
            auto t1 = tools::get_time();
            tools::timing("ctns::sweep_rdm", t0, t1);
         }
      }

   } // ctns

#endif
