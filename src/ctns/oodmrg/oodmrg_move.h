#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_disentangle.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void oodmrg_move(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const input::schedule& schd){
         const int iprt = schd.ctns.ooparams.iprt;
         // initialization
         const int dcut = schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut;
         const int& dfac = schd.ctns.ooparams.dfac;
         const int& macroiter = schd.ctns.ooparams.macroiter;
         const int& microiter = schd.ctns.ooparams.microiter;
         const double& alpha = schd.ctns.ooparams.alpha;
         const double& thrdopt = schd.ctns.ooparams.thrdopt;
         const int dmax = dfac*dcut; 
         if(iprt >= 0){
            std::cout << "\noodmrg_move: dcut=" << dcut
               << " dfac=" << dfac 
               << " macroiter=" << macroiter
               << " microiter=" << microiter
               << " alpha=" << alpha
               << " thrdopt=" << thrdopt
               << std::endl;
         }
         auto t0 = tools::get_time();

         // first optimization step
         if(iprt >= 0) std::cout << "\n### initial entanglement compression ###" << std::endl;
         reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, thrdopt, iprt);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(iprt >= 0){
               std::cout << "\n### imacro=" << imacro << ": random swap + entanglement compression ###" << std::endl;
            }

            // apply_randomlayer
            double maxdwt = reduce_entropy_single(icomb, urot, "random", dmax, alpha, iprt);

            // reduce_entropy
            reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, thrdopt, iprt);
         } // imacro 

         if(iprt >= 0){
            icomb.display_shape();
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg_move", t0, t1);
         }
      }

} // ctns

#endif
