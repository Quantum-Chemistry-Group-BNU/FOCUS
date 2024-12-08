#ifndef SWEEP_UTIL_H
#define SWEEP_UTIL_H

namespace ctns{

   // look ahead for the next fneed
   template <typename Qm, typename Tm>
      std::vector<std::string> sweep_fneed_next(const comb<Qm,Tm>& icomb,
            const std::string scratch,
            const sweep_data& sweeps,
            const int isweep,
            const int ibond,
            const bool debug){
         const auto& dots = sweeps.ctrls[isweep].dots;
         std::vector<std::string> fneed_next;
         if(ibond != sweeps.seqsize-1){
            const auto& dbond_next = sweeps.seq[ibond+1]; 
            fneed_next = icomb.topo.get_fqops(dots, dbond_next, scratch, debug);
         }else{
            if(isweep != sweeps.maxsweep-1){
               const auto& dbond_next = sweeps.seq[0];
               const auto& dots = sweeps.ctrls[isweep+1].dots; 
               fneed_next = icomb.topo.get_fqops(dots, dbond_next, scratch);
            }
         }
         return fneed_next;
      }

} // ctns

#endif
