#ifndef SWEEP_DATA_H
#define SWEEP_DATA_H

#include "../io/input.h"
#include <vector>

namespace ctns{

// computed results at a given dot	
struct dot_result{
   std::vector<double> eopt; // eopt[nstate]
   double dwt;
   int deff;
};

struct sweep_data{
   // constructor
   sweep_data(const input::schedule& schd, const std::vector<directed_bond>& sweep_seq, const int n_state){
      guess = schd.guess;
      nstate = n_state;
      seq = sweep_seq; 
      seqsize = sweep_seq.size();
      maxsweep = schd.maxsweep;
      ctrls = schd.combsweep;
      opt_result.resize(maxsweep);
      for(int i=0; i<maxsweep; i++){
	 opt_result[i].resize(seqsize+1);
	 for(int j=0; j<seqsize; j++){
	    opt_result[i][j].eopt.resize(nstate);
	 }
      }
      min_result.resize(maxsweep);
      timing.resize(maxsweep);
   }
   // print_ctrl
   void print_ctrl(const int isweep) const{
      input::combsweep_print(ctrls[isweep]);
   }
   // summary for a single sweep
   void summary(const int isweep);
public:
   bool guess;
   int maxsweep, seqsize, nstate; 
   std::vector<directed_bond> seq; // sweep bond sequence 
   std::vector<input::sweep_ctrl> ctrls; // control parameters
   std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
   std::vector<dot_result> min_result;
   std::vector<double> timing; 
};

// analysis of the current sweep (eopt,dwt,deff) and timing
void sweep_data::summary(const int isweep){
   std::cout << "\n" << tools::line_separator << std::endl;
   std::cout << "sweep_data::summary isweep=" << isweep << std::endl;
   std::cout << tools::line_separator << std::endl;
   print_ctrl(isweep);
   // print results for each dot in a single sweep
   std::vector<double> emean(seqsize,0.0);
   for(int ibond=0; ibond<seqsize; ibond++){
      auto dbond = seq[ibond];
      auto p0 = dbond.p0;
      auto p1 = dbond.p1;
      auto forward = dbond.forward;
      auto p = dbond.p;
      std::cout << " ibond=" << ibond << " bond=" << p0 << "-" << p1 
                << " forward=" << forward
                << " deff=" << opt_result[isweep][ibond].deff
                << " dwt=" << std::showpos << std::scientific << std::setprecision(2) 
	        << opt_result[isweep][ibond].dwt << std::noshowpos;
      // print energy
      std::cout << std::defaultfloat << std::setprecision(12);
      const auto& eopt = opt_result[isweep][ibond].eopt;
      int nstate = eopt.size();
      for(int j=0; j<nstate; j++){ 
         std::cout << " e[" << j << "]=" << eopt[j];
         emean[ibond] += eopt[j]; 
      }
      emean[ibond] /= nstate;
      std::cout << std::endl;
   }
   // find the minimal energy
   auto pos = std::min_element(emean.begin(), emean.end());
   auto minpos = std::distance(emean.begin(), pos);
   min_result[isweep] = opt_result[isweep][minpos];
   std::cout << "min energies at pos=" << minpos << std::endl;
   std::cout << "timing for sweep: " << std::setprecision(2) << timing[isweep] << " s" << std::endl; 
   
   // print all previous optimized results - sweep_data
   std::cout << "summary of sweep optimization up to isweep=" << isweep << std::endl;
   std::cout << "iter, dots, dcut, eps, noise | timing/s (taccum/s)" << std::endl;
   std::cout << std::scientific << std::setprecision(2);
   // print previous ctrl parameters
   double taccum = 0.0;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      auto& ctrl = ctrls[jsweep];
      taccum += timing[jsweep];
      std::cout << " " << jsweep << " "
           	<< ctrl.dots << " " << ctrl.dcut << " "
           	<< ctrl.eps  << " " << ctrl.noise << " | "
           	<< timing[jsweep] << " (" << taccum << ")" << std::endl;
   } // jsweep
   std::cout << "results: iter, dwt, energies (delta_e)" << std::endl;
   const auto& eopt = min_result[isweep].eopt;
   for(int jsweep=0; jsweep<=isweep; jsweep++){
      const auto& dwt = min_result[jsweep].dwt;
      const auto& eopt0 = min_result[jsweep].eopt;
      int nstate = eopt0.size();
      std::cout << std::setw(3) << jsweep << " ";
      std::cout << std::showpos << std::scientific << std::setprecision(2) << dwt;
      std::cout << std::noshowpos << std::defaultfloat << std::setprecision(12);
      for(int j=0; j<nstate; j++){ 
         std::cout << " e" << j << ":" 
     	      << std::defaultfloat << std::setprecision(12) << eopt0[j] << " ("
              << std::scientific << std::setprecision(2) << eopt0[j]-eopt[j] << ")";
      }
      std::cout << std::endl;
   } // jsweep
   std::cout << tools::line_separator << "\n" << std::endl;
}

} // ctns

#endif
