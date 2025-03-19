#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_urot.h"
#include "oodmrg_disentangle.h"
#include "../ctns_rcanon.h"

namespace ctns{

   // reduce entropy by carrying out multiple disentangling sweep until convergence
   template <typename Qm, typename Tm>
      double reduce_entropy_multi(comb<Qm,Tm>& icomb,
            urot_class<Tm>& urot,
            const int dmax,
            const input::params_oodmrg& ooparams){
         const int& microiter = ooparams.microiter;
         const double& alpha = ooparams.alpha;
         const double& thrdopt = ooparams.thrdopt;
         const int& iprt = ooparams.iprt;
         if(iprt >= 0){
            std::cout << "reduce_entropy_multi:" 
               << " dmax=" << dmax
               << " microiter=" << microiter
               << " alpha=" << alpha
               << " thrdopt=" << thrdopt
               << std::endl;
         }
         auto t0 = tools::get_time();

         // initialization
         double s_init = rcanon_entropysum(icomb, alpha); // record the input entropy
         double s_old = s_init, s_new = s_init, s_diff;
         
         // optimization
         double maxdwt = -1.0;
         bool ifconv = false;
         for(int imicro=0; imicro<microiter; imicro++){
            if(iprt > 0){
               std::cout << "=== imicro=" << imicro << " ===" << std::endl;
            }
            // optimize
            std::string scheme = ooparams.swap? "swap" : "opt"; 
            double imaxdwt = reduce_entropy_single(icomb, urot, scheme, dmax, ooparams);
            maxdwt = std::max(maxdwt, imaxdwt);
            s_new = rcanon_entropysum(icomb, alpha);
            s_diff = s_new - s_old;
            if(iprt > 0){
               std::cout << "result:" << std::scientific
                  << " s[old]=" << s_old << " s[new]=" << s_new 
                  << " s[diff]=" << s_diff << " imaxdwt=" << imaxdwt 
                  << std::endl;
            }
            // check convergence
            if(std::abs(s_diff) < thrdopt){
               if(iprt >= 0){
                  std::cout << "converge in " << (imicro+1) << " iterations:"
                     << std::setprecision(4) << " s[init]=" << s_init 
                     << " s[new]=" << s_new << " maxdwt=" << maxdwt
                     << std::endl;
               }
               ifconv = true; 
               break;          
            }else{
               s_old = s_new;
            }
         } // imicro
         if(not ifconv){
            std::cout << "Warning: reduce_entropy_multi does not converge in microiter="
               << microiter << std::endl;
         }
         
         if(iprt >= 0){
            auto t1 = tools::get_time();
            tools::timing("ctns::reduce_entropy_multi", t0, t1);
         }
         return s_new;
      }

   template <typename Qm, typename Tm>
      double oodmrg_move(comb<Qm,Tm>& icomb,
            urot_class<Tm>& urot,
            const input::schedule& schd){
         const int iprt = schd.ctns.ooparams.iprt;
         const int& macroiter = schd.ctns.ooparams.macroiter;
         const int& microiter = schd.ctns.ooparams.microiter;
         const double& alpha = schd.ctns.ooparams.alpha;
         const int dcut = schd.ctns.maxsweep>0? schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut : icomb.get_dmax();
         const int dfac = schd.ctns.ooparams.dfac;
         const int dmax = dfac*dcut; 
         if(iprt >= 0){
            std::cout << "\noodmrg_move:"
               << " macroiter=" << macroiter
               << " microiter=" << microiter
               << " alpha=" << alpha
               << " dfac=" << dfac 
               << " dcut=" << dcut
               << " dmax=" << dmax 
               << std::endl;
         }
         auto t0 = tools::get_time();

         // save the initial u0 
         auto u0 = urot;

         // first optimization step
         if(iprt >= 0) std::cout << "\n### initial entanglement compression ###" << std::endl;
         double s_old = reduce_entropy_multi(icomb, urot, dmax, schd.ctns.ooparams);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(iprt >= 0){
               std::cout << "\n### imacro=" << imacro 
                  << ": random swap + entanglement compression ###"
                  << std::endl;
            }
            // apply_randomlayer
            double maxdwt = reduce_entropy_single(icomb, urot, "randomswap", dmax, schd.ctns.ooparams);
            // reduce_entropy
            double s_new = reduce_entropy_multi(icomb, urot, dmax, schd.ctns.ooparams);
            if(iprt >= 0){
               std::cout << "imacro=" << imacro 
                  << std::scientific << std::setprecision(4)
                  << " s[old]=" << s_old
                  << " s[new]=" << s_new
                  << std::endl;
            }
         } // imacro 

         // change the last site of MPS to identity for later optimization
         rcanon_lastdots(icomb);

         // urot = u0*U => U = u0.H()*urot
         auto umove_0 = linalg::xgemm("C","N",u0.umat[0],urot.umat[0]);
         auto umove_1 = linalg::xgemm("C","N",u0.umat[1],urot.umat[1]);
         double u_dev = (check_identityMatrix(umove_0) + check_identityMatrix(umove_1))/2.0;
         if(iprt >= 0){
            std::cout << "\noodmrg_move: |U[move]-I|_F=" 
               << std::scientific << std::setprecision(2) << u_dev 
               << std::endl;
            icomb.display_shape();
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg_move", t0, t1);
         }
         return u_dev;
      }

} // ctns

#endif
