#ifndef SYMBOLIC_FORMULAE_RENORM_H
#define SYMBOLIC_FORMULAE_RENORM_H

#include "symbolic_task.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"
#include "../core/tools.h"

namespace ctns{

   // see oper_renorm_kernel.h
   template <typename Tm>
      renorm_tasks<Tm> gen_formulae_renorm(const std::string& oplist,
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
         const int print_level = 1;

         renorm_tasks<Tm> formulae;
         size_t idx = 0;

         // opC
         if(oplist.find('C') != std::string::npos){
            counter["C"] = 0;	   
            auto info = oper_combine_opC(cindex1, cindex2);
            for(const auto& pr : info){
               int index = pr.first, iformula = pr.second;
               auto opC = symbolic_normxwf_opC<Tm>(block1, block2, index, iformula);
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
                  auto opA = symbolic_normxwf_opA<Tm>(block1, block2, index, iformula, ifkr);
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
                  auto opB = symbolic_normxwf_opB<Tm>(block1, block2, index, iformula, ifkr, ifDop);
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
            auto pindex = oper_index_opP(krest, ifkr);    
            for(const auto& index : pindex){
               int iproc = distribute2('P',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opP = symbolic_compxwf_opP<Tm>(block1, block2, cindex1, cindex2,
                        int2e, index, isym, ifkr);
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
            auto qindex = oper_index_opQ(krest, ifkr); 
            for(const auto& index : qindex){
               int iproc = distribute2('Q',ifkr,size,index,sorb);
               if(iproc == rank){
                  auto opQ = symbolic_compxwf_opQ<Tm>(block1, block2, cindex1, cindex2,
                        int2e, index, isym, ifkr);
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
               auto opS = symbolic_compxwf_opS<Tm>(oplist1, oplist2, block1, block2, cindex1, cindex2,
                     int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
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
            auto opH = symbolic_compxwf_opH<Tm>(oplist1, oplist2, block1, block2, cindex1, cindex2,
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
            auto opI = symbolic_normxwf_opI<Tm>(block1, block2);  
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
               auto opD = symbolic_normxwf_opD<Tm>(block1, block2, index, iformula);
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
                  auto opM = symbolic_normxwf_opM<Tm>(block1, block2, index, iformula, ifkr);
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

   template <typename Tm>
      renorm_tasks<Tm> symbolic_formulae_renorm(const std::string superblock,
            const integral::two_body<Tm>& int2e,
            const oper_dict<Tm>& qops1,
            const oper_dict<Tm>& qops2,
            const oper_dict<Tm>& qops,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool ifdistc,
            const bool debug=false){
         auto t0 = tools::get_time();
         const std::string block1 = superblock.substr(0,1);
         const std::string block2 = superblock.substr(1,2);
         const auto& cindex1 = qops1.cindex;
         const auto& cindex2 = qops2.cindex;
         const int isym = qops.isym;
         const bool ifkr = qops.ifkr;
         const bool ifhermi = qops.ifhermi;
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_renorm"
                  << " qops.oplist=" << qops.oplist
                  << " mpisize=" << size
                  << " fname=" << fname
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "ctns::symbolic_formulae_renorm"
               << " isym=" << isym
               << " ifkr=" << ifkr
               << " block1=" << block1
               << " block2=" << block2
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }
         // generation of renorm
         std::map<std::string,int> counter;
         auto rformulae = gen_formulae_renorm(qops.oplist,
               qops1.oplist,qops2.oplist,block1,block2,
               cindex1,cindex2,qops.krest,isym,ifkr,ifhermi,
               int2e,qops.sorb,size,rank,ifdist1,ifdistc,ifsave,counter);
         // reorder if necessary
         if(sort_formulae){
            std::map<std::string,int> dims = {{block1,qops1.qket.get_dimAll()},
               {block2,qops2.qket.get_dimAll()}};
            rformulae.sort(dims);
         }
         if(ifsave){
            std::cout << "\nSUMMARY:" << std::endl;
            rformulae.display("total");
            qops1.print("qops1",2);
            qops2.print("qops2",2);
            qops.print("qops",2);
            std::cout.rdbuf(backup); // restore cout's original streambuf
            file.close();
         }
         if(rank == 0 and debug){
            auto t1 = tools::get_time();
            int size = rformulae.size();
            tools::timing("symbolic_formulae_renorm with size="+std::to_string(size), t0, t1);
         }
         return rformulae;
      }

} // ctns

#endif
