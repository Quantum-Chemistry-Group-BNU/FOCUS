#ifndef CTNS_OODMRG_MOVE_H
#define CTNS_OODMRG_MOVE_H

#include "oodmrg_disentangle.h"

namespace ctns{

   // reduce entropy by carrying out multiple disentangling sweep until convergence
   template <typename Qm, typename Tm>
      void reduce_entropy_multi(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const int dmax,
            const input::params_orbopt& ooparams){
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
         auto icomb0 = icomb;
         double totaldiff = 0.0;
         bool ifconv = false;
         double s_init = sum_of_entropy(icomb, alpha); // record the input entropy
         double s_old = s_init;
         double maxdwt = -1.0;

         // optimization
         for(int imicro=0; imicro<microiter; imicro++){
            if(iprt > 0){
               std::cout << "=== imicro=" << imicro << " ===" << std::endl;
            }
            // optimize
            double imaxdwt = reduce_entropy_single(icomb, urot, "opt", dmax, ooparams);
            maxdwt = std::max(maxdwt,imaxdwt);
            double s_new = sum_of_entropy(icomb, alpha);
            double s_diff = s_new - s_old;
            if(iprt > 0){
               auto smat = get_Smat(icomb,icomb0);
               std::cout << "result:" << std::scientific
                  << " s[old]=" << s_old
                  << " s[new]=" << s_new << " s[diff]=" << s_diff
                  << " imaxdwt=" << imaxdwt 
                  << " <MPS[0]|MPS[new]>=" << smat(0,0)
                  << std::endl;
            }
            // check convergence
            if(std::abs(s_diff) < thrdopt){
               if(iprt >= 0){
                  std::cout << "converge in "
                     << (imicro+1) << " iterations:"
                     << std::setprecision(4)
                     << " s[init]=" << s_init 
                     << " s[new]=" << s_new
                     << " maxdwt=" << maxdwt
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
      }

   template <typename Qm, typename Tm>
      double oodmrg_move(comb<Qm,Tm>& icomb,
            linalg::matrix<Tm>& urot,
            const input::schedule& schd){
         const int iprt = schd.ctns.ooparams.iprt;
         const int& macroiter = schd.ctns.ooparams.macroiter;
         const int& microiter = schd.ctns.ooparams.microiter;
         const double& alpha = schd.ctns.ooparams.alpha;
         const int dcut = schd.ctns.ctrls[schd.ctns.maxsweep-1].dcut;
         const int dfac = schd.ctns.ooparams.dfac;
         const int dmax = dfac*dcut; 
         if(iprt >= 0){
            std::cout << "\noodmrg_move: dcut=" << dcut
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
         reduce_entropy_multi(icomb, urot, dmax, schd.ctns.ooparams);

         // start subsequent optimization
         for(int imacro=0; imacro<macroiter; imacro++){
            if(iprt >= 0){
               std::cout << "\n### imacro=" << imacro 
                  << ": random swap + entanglement compression ###"
                  << std::endl;
            }
            // apply_randomlayer
            double maxdwt = reduce_entropy_single(icomb, urot, "random", dmax, schd.ctns.ooparams);
            // reduce_entropy
            reduce_entropy_multi(icomb, urot, dmax, schd.ctns.ooparams);
         } // imacro 

         // change the last site of MPS to identity for later optimization
         rcanon_lastdot(icomb);

         // urot = u0*U => U = u0.H()*urot
         int norb = u0.rows();
         auto udiff = linalg::xgemm("C","N",u0,urot) - linalg::identity_matrix<Tm>(norb);
         double u_diff = linalg::xnrm2(udiff.size(), udiff.data());
         if(iprt >= 0){
            std::cout << "\noodmrg_move: |U-I|_F=" << u_diff << std::endl;
            icomb.display_shape();
            auto t1 = tools::get_time();
            tools::timing("ctns::oodmrg_move", t0, t1);
         }
         return u_diff;
      }

} // ctns

#endif
