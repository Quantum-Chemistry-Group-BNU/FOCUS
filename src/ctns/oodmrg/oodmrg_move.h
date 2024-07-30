#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_disentangle.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void oodmrg_move(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const input::schedule& schd,
            const bool debug=true){
         // initialization
         const int dcut = schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut;
         const int& dfac = schd.ctns.ooparams.dfac;
         const int& macroiter = schd.ctns.ooparams.macroiter;
         const int& microiter = schd.ctns.ooparams.microiter;
         const double& alpha = schd.ctns.ooparams.alpha;
         const int dmax = dfac*dcut; 
         if(debug){
            std::cout << "oodmrg_move: dcut=" << dcut
               << " dfac=" << dfac 
               << " macroiter=" << macroiter
               << " microiter=" << microiter
               << " alpha=" << alpha
               << std::endl;
         }

         // first optimization step
         reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, debug);
         exit(1);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(debug){
               std::cout << "\n### imacro=" << imacro << " ###" << std::endl;
            }

            // apply_randomlayer
            //apply_randomlayer(icomb, urot);

            // reduce_entropy
            reduce_entropy_multi(icomb, urot, dmax, microiter, alpha, debug);

         } // imacro 
         exit(1);
      }

} // ctns

#endif
