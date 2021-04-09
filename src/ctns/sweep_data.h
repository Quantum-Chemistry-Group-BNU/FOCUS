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
   sweep_data(const input::schedule& schd, const int seq_size){
      seqsize = seq_size; 
      maxsweep = schd.maxsweep;
      ctrls = schd.combsweep;
      opt_result.resize(maxsweep);
      for(int i=0; i<maxsweep; i++){
	 opt_result[i].resize(seqsize+1);
      }
      timing.resize(maxsweep);
   } 
public:
   int maxsweep, seqsize; 
   std::vector<input::sweep_ctrl> ctrls;
   std::vector<std::vector<dot_result>> opt_result; // (maxsweep,seqsize) 
   std::vector<double> timing; 
};

} // ctns

#endif
