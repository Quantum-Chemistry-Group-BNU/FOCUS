#ifndef RDM_DOT_H
#define RDM_DOT_H

#include "../oper_dot.h"
#include "../sadmrg/oper_dot_su2.h"

namespace ctns{

   // Abelian case
   template <bool ifab, typename Tm>
      void rdm_init_dot(const int order,
            const bool is_same,
            const int sorb,
            qoper_dict<ifab,Tm>& qops,
            const int isym,
            const bool ifkr,
            const int kp,
            const int size,
            const int rank){
         // setup basic information
         qops.sorb = sorb;
         qops.isym = isym;
         qops.ifkr = ifkr;
         qops.cindex.push_back(2*kp);
         if(!ifkr) qops.cindex.push_back(2*kp+1);
         // rest of spatial orbital indices
         for(int k=0; k<sorb/2; k++){
            if(k == kp) continue;
            qops.krest.push_back(k);
         }
         auto qphys = get_qbond_phys(isym);
         qops.qbra = qphys;
         qops.qket = qphys;
         if(order == 1){
            if(is_same){
               qops.oplist = "ICB";
            }else{
               qops.oplist = "ICBD";
            }
         }else if(order >= 2){
            if(is_same){
               qops.oplist = "ICABTF";
            }else{
               qops.oplist = "ICABTFDM";
            }
         }
         // initialize memory
         qops.ifhermi = false; 
         qops.init(true);
         // compute local operators on dot
         oper_dot_opI(qops);
         oper_dot_opC(qops, kp);
         oper_dot_opB(qops, kp);
         if(order >= 2){
            oper_dot_opA(qops, kp);
            oper_dot_opT(qops, kp);
            oper_dot_opF(qops, kp);
         }
         // for icomb != icomb2
         if(!is_same){
            oper_dot_opD(qops, kp);
            if(order >= 2){
               oper_dot_opM(qops, kp);
            }
         }
      }

} // ctns

#endif
