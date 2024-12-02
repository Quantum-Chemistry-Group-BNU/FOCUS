#ifndef SYMBOLIC_FORMULAE_RENORM_SU2_H
#define SYMBOLIC_FORMULAE_RENORM_SU2_H

#include "../../core/tools.h"
#include "../symbolic_task.h"
#include "symbolic_normxwf_su2.h"
#include "symbolic_compxwf_su2.h"

namespace ctns{

   template <typename Tm>
      renorm_tasks<Tm> gen_formulae_renorm_su2(const std::string& oplist,
            const std::string& oplist1,
            const std::string& oplist2,
            const std::string& block1,
            const std::string& block2,
            const std::vector<int>& cindex1,
            const std::vector<int>& cindex2,
            const std::vector<int>& krest,
            const int isym,
            const bool ifkr,
            const bool ifhermi,
            const integral::two_body<Tm>& int2e,
            const int sorb,
            const int size,
            const int rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            std::map<std::string,int>& counter){
         assert(isym == 3);
         const int print_level = 1;

         renorm_tasks<Tm> formulae;
         size_t idx = 0;

         // opC
         if(oplist.find('C') != std::string::npos){
            counter["C"] = 0;	   
            auto info = oper_combine_opC(cindex1, cindex2);
            for(const auto& pr : info){
               int index = pr.first, iformula = pr.second;
               auto opC = symbolic_normxwf_opC_su2<Tm>(block1, block2, index, iformula);
               formulae.append(std::make_tuple('C', index, opC));
               counter["C"] += opC.size();
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  opC.display("opC["+std::to_string(index)+"]", print_level);
               }
            }
         }
         // opA
         if(oplist.find('A') != std::string::npos){
            counter["A"] = 0;
            auto ainfo = oper_combine_opA(cindex1, cindex2, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first, iformula = pr.second;
               int iproc = distribute2('A',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opA = symbolic_normxwf_opA_su2<Tm>(block1, block2, index, iformula);
                  formulae.append(std::make_tuple('A', index, opA));
                  counter["A"] += opA.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     opA.display("opA["+std::to_string(index)+"]", print_level);
                  }
               }
            }
         }
         // opB
         if(oplist.find('B') != std::string::npos){
            bool ifDop = oplist.find('D') != std::string::npos; 
            counter["B"] = 0;
            auto binfo = oper_combine_opB(cindex1, cindex2, ifkr, ifhermi);
            for(const auto& pr : binfo){
               int index = pr.first, iformula = pr.second;
               int iproc = distribute2('B',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opB = symbolic_normxwf_opB_su2<Tm>(block1, block2, index, iformula, ifDop);
                  formulae.append(std::make_tuple('B', index, opB));
                  counter["B"] += opB.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     opB.display("opB["+std::to_string(index)+"]", print_level);
                  }
               }
            }
         }
         // opP
         if(oplist.find('P') != std::string::npos){
            counter["P"] = 0;	
            auto pindex = oper_index_opP(krest, ifkr, isym);    
            for(const auto& index : pindex){
               int iproc = distribute2('P',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opP = symbolic_compxwf_opP_su2<Tm>(block1, block2, cindex1, cindex2,
                        int2e, index);
                  formulae.append(std::make_tuple('P', index, opP));
                  counter["P"] += opP.size();	   
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     opP.display("opP["+std::to_string(index)+"]", print_level);
                  }
               }
            }
         }
         // opQ
         if(oplist.find('Q') != std::string::npos){
            counter["Q"] = 0;
            auto qindex = oper_index_opQ(krest, ifkr, isym); 
            for(const auto& index : qindex){
               int iproc = distribute2('Q',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opQ = symbolic_compxwf_opQ_su2<Tm>(block1, block2, cindex1, cindex2,
                        int2e, index);
                  formulae.append(std::make_tuple('Q', index, opQ));
                  counter["Q"] += opQ.size();	   
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     opQ.display("opQ["+std::to_string(index)+"]", print_level);
                  }
               }
            }
         }
         // opS
         if(oplist.find('S') != std::string::npos){
            counter["S"] = 0;
            auto sindex = oper_index_opS(krest, ifkr); 
            for(const auto& index : sindex){
               auto opS = symbolic_compxwf_opS_su2<Tm>(oplist1, oplist2, block1, block2, cindex1, cindex2,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               // opS can be empty for ifdist1=true
               if(opS.size() == 0) continue;
               formulae.append(std::make_tuple('S', index, opS));
               counter["S"] += opS.size();	   
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  opS.display("opS["+std::to_string(index)+"]", print_level);
               }
            }
         }
         // opH
         if(oplist.find('H') != std::string::npos){
            counter["H"] = 0;	   
            auto opH = symbolic_compxwf_opH_su2<Tm>(oplist1, oplist2, block1, block2, cindex1, cindex2,
                  int2e, ifkr, sorb, size, rank, ifdist1, ifdistc);
            // opH can be empty for ifdist1=true
            if(opH.size() > 0){
               formulae.append(std::make_tuple('H', 0, opH));
               counter["H"] += opH.size();	   
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  opH.display("opH", print_level);
               }
            }
         }

         // --- for RDM calculations ---
         // opI
         if(oplist.find('I') != std::string::npos){
            counter["I"] = 0;	   
            auto opI = symbolic_normxwf_opI_su2<Tm>(block1, block2);
            formulae.append(std::make_tuple('I', 0, opI));
            counter["I"] += opI.size();	   
            if(ifsave){
               std::cout << "idx=" << idx++;
               opI.display("opI", print_level);
            }
         }
         // opD
         if(oplist.find('D') != std::string::npos){
            counter["D"] = 0;	   
            auto info = oper_combine_opC(cindex1, cindex2);
            for(const auto& pr : info){
               int index = pr.first, iformula = pr.second;
               auto opD = symbolic_normxwf_opD_su2<Tm>(block1, block2, index, iformula);
               formulae.append(std::make_tuple('D', index, opD));
               counter["D"] += opD.size();
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  opD.display("opD["+std::to_string(index)+"]", print_level);
               }
            }
         }
         // opM
         if(oplist.find('M') != std::string::npos){
            counter["M"] = 0;
            auto ainfo = oper_combine_opA(cindex1, cindex2, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first, iformula = pr.second;
               int iproc = distribute2('M',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opM = symbolic_normxwf_opM_su2<Tm>(block1, block2, index, iformula);
                  formulae.append(std::make_tuple('M', index, opM));
                  counter["M"] += opM.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     opM.display("opM["+std::to_string(index)+"]", print_level);
                  }
               }
            }
         }
         return formulae;
      }

} // ctns

#endif
