#ifndef RDM_DOT_H
#define RDM_DOT_H

#include "../oper_dot.h"
#include "../sadmrg/oper_dot_su2.h"

namespace ctns{

   // Abelian case
   template <bool ifab, typename Tm>
      void rdm_init_dot(const int order,
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
            qops.oplist = "CB";
         }else if(order == 2){
            //qops.oplist = "CAB...34body...";
         }else{
            std::cout << "error: rdm_init_dot does not support order=" << order << std::endl;
            exit(1);
         }
         // initialize memory
         qops.init(true);
         // compute local operators on dot
         oper_dot_opC(qops, kp);
         oper_dot_opB(qops, kp);
      }

} // ctns

#endif
