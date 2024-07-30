#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_entropy.h"

namespace ctns{

   template <typename Qm, typename Tm>
      void reduce_entropy_single(comb<Qm,Tm>& icomb_new,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const double alpha,
            const bool debug){
         if(debug){
            std::cout << "reduce_entropy_single" << std::endl;
         }
      }

   template <typename Qm, typename Tm>
      void reduce_entropy_multi(comb<Qm,Tm>& icomb_new,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const int microiter,
            const double alpha,
            const bool debug){
         const double thresh = 1.e-8;
         double totaldiff = 0.0;
         bool ifconv = false;
         double s_old = sum_of_entropy(icomb_new, alpha);

         std::cout << "s_old=" << s_old << std::endl;
         exit(1);
         for(int imicro=0; imicro<microiter; imicro++){
            if(debug){
               std::cout << "\n=== imicro=" << imicro << " ===" << std::endl;
            }
            reduce_entropy_single(icomb_new, urot, dmax, alpha, debug);
            double s_new = sum_of_entropy(icomb_new, alpha);
            double diff = s_new - s_old;
            if(debug){
               std::cout << "imicro=" << imicro << " s_old=" << s_old
                  << " s_new=" << s_new << " diff=" << diff
                  << std::endl;
            }
            if(std::abs(diff) < thresh){
               if(debug) std::cout << "reduce_entropy_multi converges!" << std::endl;
               ifconv = true; 
               break;          
            }else{
               s_old = s_new;
            }
         } // imicro
         if(not ifconv){
            std::cout << "Warning: reduce_entropy_multi does not converge!" << std::endl;
         }
      }

   template <typename Qm, typename Tm>
      void oodmrg_move(comb<Qm,Tm>& icomb_new,
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
         reduce_entropy_multi(icomb_new, urot, dmax, microiter, alpha, debug);
         exit(1);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(debug){
               std::cout << "\n### imacro=" << imacro << " ###" << std::endl;
            }

            // apply_randomlayer
            //apply_randomlayer(icomb_new, urot);

            // reduce_entropy
            reduce_entropy_multi(icomb_new, urot, dmax, microiter, alpha, debug);

         } // imacro 
         exit(1);
      }

} // ctns

#endif
