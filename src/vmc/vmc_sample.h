#ifndef VMC_SAMPLE_H
#define VMC_SAMPLE_H

#include "ansatz.h"

namespace vmc{

   fock::onspace get_sample(BaseAnsatz& wavefun,
         const int nsample, 
         const fock::onstate& seed){
      std::cout << "\nvmc::get_sample" << std::endl; 
      fock::onspace space(nsample);
      fock::onstate state(seed);
      double lnpsiR = wavefun.lnpsi(state).real();
      int no = state.nelec(), k = state.size(), nv = k - no;
      std::vector<int> olst(no), vlst(nv);
      // random singles
      int nsingles = no*nv;
      std::vector<double> weights(nsingles,1.0/nsingles); 
      std::discrete_distribution<> dist(weights.begin(),weights.end());
      std::uniform_real_distribution<double> udist(0,1);
      // start Markov-Chain Monte-Carlo
      // https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm
      int noff = 1000;
      int maxcycle = nsample + noff;
      int naccept = 0;
      for(int k=0; k<maxcycle; k++){
         // generate a new configuration from seed
         state.get_olst(olst.data()); 
         state.get_vlst(vlst.data());
         int ia = dist(tools::generator);
         int ix = ia%no, ax = ia/no;
         int i = olst[ix], a = vlst[ax];
         fock::onstate state1(state);
         state1[i] = 0;
         state1[a] = 1;
         double lnpsi1R = wavefun.lnpsi(state1).real();
         double prob_ratio = std::exp(2.0*(lnpsi1R - lnpsiR));
         double paccept = std::min(1.0,prob_ratio);
         double u = udist(tools::generator);
         /*
         std::cout << std::setprecision(10);
         std::cout << "state=" << state << " lnpsiR=" << lnpsiR << std::endl;
         std::cout << "state1=" << state1 << " lnpsi1R=" << lnpsi1R << std::endl; 
         std::cout << "prob_ratio=" << prob_ratio << std::endl;
         */
         if(u <= paccept){
            state = state1;
            lnpsiR = lnpsi1R;
            if(k >= noff) naccept += 1;
         }
         if(k >= noff){
            space[k-noff] = state;
            /*
            std::cout << "i=" << k-noff 
                      << " " << prob_ratio 
                      << " " << u 
                      << " " << state 
                      << std::endl;
            */
         }
      }
      std::cout << " acceptance ratio =" << naccept/double(nsample) << std::endl; 
      return space;
   }

} // vmc

#endif
