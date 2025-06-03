#ifndef CTNS_SCRAMBLE_H
#define CTNS_SCRAMBLE_H

#include "oodmrg/oodmrg_move.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void random_givens(comb<Qm,Tm>& icomb, 
            const input::schedule& schd){
         int size = 1, rank = 0;
#ifndef SERIAL
         size = icomb.world.size();
         rank = icomb.world.rank();
#endif  
         const int dcut = schd.ctns.maxsweep>0? schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut : icomb.get_dmax();
         const int dfac = schd.ctns.ooparams.dfac;
         if(rank == 0){
            std::cout << "\nctns::random_givens" 
               << " depth=" << schd.ctns.depth 
               << " dcut=" << dcut
               << " dfac=" << dfac
               << std::endl;
         }
         auto t0 = tools::get_time();
         
         auto icomb0 = icomb;
         const int norb = icomb.get_nphysical();
         urot_class<Tm> urot(false, norb);
         for(int i=0; i<schd.ctns.depth; i++){
            if(rank == 0){ 
               std::cout << "### i=" << i << " : random layer of Given rotation ###" << std::endl;
            }
            const int dmax = (i==schd.ctns.depth-1)? dcut : dfac*dcut; 
            double maxdwt = reduce_entropy_single(icomb, urot, "random", dmax, schd.ctns.ooparams);
            if(rank == 0){
               std::cout << " maxdwt=" << maxdwt << std::endl;
            } 
         }

         // check ovlp
         if(rank == 0){
            icomb.display_shape();
            auto smat = get_Smat(icomb, icomb);
            std::cout << "\nfinal results of ctns::random_givens:" << std::endl;
            smat.print("<Psi|Psi>", 10);
            auto smat0 = get_Smat(icomb, icomb0);
            smat0.print("<Psi|Psi0>", 10);
            auto t1 = tools::get_time();
            tools::timing("ctns::random_givens", t0, t1);
         }
      }

} // ctns

#endif
