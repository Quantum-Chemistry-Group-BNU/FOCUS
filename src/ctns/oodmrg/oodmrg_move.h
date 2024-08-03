#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_disentangle.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void oodmrg_move(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const input::schedule& schd){
         const bool debug = schd.ctns.ooparams.iprt > 0;
         // initialization
         const int dcut = schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut;
         const int& dfac = schd.ctns.ooparams.dfac;
         const int& macroiter = schd.ctns.ooparams.macroiter;
         const int& microiter = schd.ctns.ooparams.microiter;
         const double& alpha = schd.ctns.ooparams.alpha;
         const double& thrdopt = schd.ctns.ooparams.thrdopt;
         const int dmax = dfac*dcut; 
         if(debug){
            std::cout << "oodmrg_move: dcut=" << dcut
               << " dfac=" << dfac 
               << " macroiter=" << macroiter
               << " microiter=" << microiter
               << " alpha=" << alpha
               << " thrdopt=" << thrdopt
               << std::endl;
         }

         // first optimization step
         reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, thrdopt, debug);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(debug){
               std::cout << "\n### imacro=" << imacro << " ###" << std::endl;
            }

            // apply_randomlayer
            double maxdwt = reduce_entropy_single(icomb, urot, "random", dmax, alpha, debug);
            
            // reduce_entropy
            reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, thrdopt, debug);
         } // imacro 

         icomb.display_shape();
         exit(1);
      }

} // ctns

#endif
