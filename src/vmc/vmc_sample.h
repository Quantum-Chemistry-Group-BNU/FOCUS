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
      double prob = std::pow(std::abs(wavefun.psi(state)),2);
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
         double prob1 = std::pow(std::abs(wavefun.psi(state1)),2);
         double paccept = std::min(1.0,prob1/prob);
         double u = udist(tools::generator);
         if(u <= paccept){
            state = state1;
            prob = prob1;
            if(k >= noff) naccept += 1;
         }
         if(k >= noff){
            space[k-noff] = state;
         }
      }
      std::cout << " acceptance ratio =" << naccept/double(nsample) << std::endl; 
      return space;
   }

} // vmc

#endif
