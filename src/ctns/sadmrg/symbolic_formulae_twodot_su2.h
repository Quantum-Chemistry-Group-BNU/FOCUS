#ifndef SYMBOLIC_FORMULAE_TWODOT_SU2_H
#define SYMBOLIC_FORMULAE_TWODOT_SU2_H

#include "../../core/tools.h"
#include "../oper_dict.h"
#include "../symbolic_oper.h"
#include "symbolic_normxwf_su2.h"
#include "symbolic_compxwf_su2.h"

namespace ctns{

   template <typename Tm>
      symbolic_task<Tm> gen_formulae_twodot_su2(const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c1,
            const std::vector<int>& cindex_c2,
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            std::map<std::string,int>& counter){
         const int print_level = 1;
         int slc1 = cindex_l.size() + cindex_c1.size();
         int sc2r = cindex_c2.size() + cindex_r.size();
         const bool ifNC = (slc1 <= sc2r);

         symbolic_task<Tm> formulae;
         int idx = 0;

         // Local terms:
         // H[lc1]
         auto Hlc1 = symbolic_compxwf_opH_su2<Tm>("l", "c1", cindex_l, cindex_c1, 
               ifkr, int2e.sorb, size, rank, ifdist1);
         counter["H1"] = Hlc1.size();
         if(Hlc1.size() > 0){
            auto op2 = symbolic_prod<Tm>(symbolic_oper("c2",'I',0),symbolic_oper("r",'I',0));
            op2.ispins.push_back(std::make_tuple(0,0,0));
            symbolic_task<Tm> Ic2r;
            Ic2r.append(op2);
            auto Hlc1_Ic2r = Hlc1.outer_product(Ic2r);
            Hlc1_Ic2r.append_ispins(std::make_tuple(0,0,0));
            formulae.join(Hlc1_Ic2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++; 
               Hlc1_Ic2r.display("Hlc1_Ic2r", print_level);
            }
         }
         // H[c2r]
         auto Hc2r = symbolic_compxwf_opH_su2<Tm>("c2", "r", cindex_c2, cindex_r, 
               ifkr, int2e.sorb, size, rank, ifdist1);
         counter["H2"] = Hc2r.size();
         if(Hc2r.size() > 0){
            auto op1 = symbolic_prod<Tm>(symbolic_oper("l",'I',0),symbolic_oper("c1",'I',0));
            op1.ispins.push_back(std::make_tuple(0,0,0));
            symbolic_task<Tm> Ilc1;
            Ilc1.append(op1);
            auto Ilc1_Hc2r = Ilc1.outer_product(Hc2r);
            Ilc1_Hc2r.append_ispins(std::make_tuple(0,0,0)); 
            formulae.join(Ilc1_Hc2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Ilc1_Hc2r.display("Ilc1_Hc2r", print_level);
            }
         }

         // One-index terms:
         // 3. sum_p1 p1^+[LC1]*Sp1^[C2R] + h.c.
         counter["CS"] = 0;
         auto infoC1 = oper_combine_opC(cindex_l, cindex_c1);
         for(const auto& pr : infoC1){
            int index = pr.first;
            int iformula = pr.second;
            // p1^L1C1+*Sp1^C2R & -p1^L1C1*Sp1^C2R+
            auto Clc1 = symbolic_normxwf_opC_su2<Tm>("l", "c1", index, iformula);
            auto Sc2r = symbolic_compxwf_opS_su2<Tm>("c2", "r", cindex_c2, cindex_r,
                  int2e, index, ifkr, size, rank, ifdist1, ifdistc);
            if(Sc2r.size() == 0) continue;
            auto Clc1_Sc2r = Clc1.outer_product(Sc2r);
            Clc1_Sc2r.scale(std::sqrt(2.0));
            Clc1_Sc2r.append_ispins(std::make_tuple(1,1,0));
            formulae.join(Clc1_Sc2r);
            counter["CS"] += Clc1_Sc2r.size();
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Clc1_Sc2r.display("Clc1_Sc2r["+std::to_string(index)+"]", print_level);
            }
         }
         // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
         counter["SC"] = 0;
         auto infoC2 = oper_combine_opC(cindex_c2, cindex_r);
         for(const auto& pr : infoC2){
            int index = pr.first;
            int iformula = pr.second;
            // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
            auto Slc1 = symbolic_compxwf_opS_su2<Tm>("l", "c1", cindex_l, cindex_c1,
                  int2e, index, ifkr, size, rank, ifdist1, ifdistc);
            if(Slc1.size() == 0) continue;
            auto Cc2r = symbolic_normxwf_opC_su2<Tm>("c2", "r", index, iformula);
            auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
            Slc1_Cc2r.scale(std::sqrt(2.0));
            Slc1_Cc2r.append_ispins(std::make_tuple(1,1,0));
            formulae.join(Slc1_Cc2r);
            counter["SC"] += Slc1_Cc2r.size();
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Slc1_Cc2r.display("Slc1_Cc2r["+std::to_string(index)+"]", print_level);
            }
         }

         // Two-index terms:
         if(ifNC){

            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["AP"] = 0;
            auto ainfo = oper_combine_opA(cindex_l, cindex_c1, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2;
               int iformula = pr.second;
               int iproc = distribute2('A',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Alc1 = symbolic_normxwf_opA_su2<Tm>("l", "c1", index, iformula);
                  auto Pc2r = symbolic_compxwf_opP_su2<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index);
                  auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  Alc1_Pc2r.scale(fac);
                  Alc1_Pc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Alc1_Pc2r);
                  counter["AP"] += Alc1_Pc2r.size();
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Alc1_Pc2r.display("Alc1_Pc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["BQ"] = 0;
            auto binfo = oper_combine_opB(cindex_l, cindex_c1, ifkr);
            for(const auto& pr : binfo){
               int index = pr.first;
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               int iformula = pr.second;
               int iproc = distribute2('B',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  auto Blc1 = symbolic_normxwf_opB_su2<Tm>("l", "c1", index, iformula);
                  auto Qc2r = symbolic_compxwf_opQ_su2<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index);
                  auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  Blc1_Qc2r.scale(fac);
                  Blc1_Qc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Blc1_Qc2r);
                  counter["BQ"] += Blc1_Qc2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Blc1_Qc2r.display("Blc1_Qc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }

         }else{

            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["PA"] = 0;
            auto ainfo = oper_combine_opA(cindex_c2, cindex_r, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2;
               int iformula = pr.second;
               int iproc = distribute2('A',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Plc1 = symbolic_compxwf_opP_su2<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index);
                  auto Ac2r = symbolic_normxwf_opA_su2<Tm>("c2", "r", index, iformula);
                  auto Plc1_Ac2r = Plc1.outer_product(Ac2r);
                  double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
                  Plc1_Ac2r.scale(fac);
                  Plc1_Ac2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Plc1_Ac2r);
                  counter["PA"] += Plc1_Ac2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Plc1_Ac2r.display("Plc1_Ac2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["QB"] = 0;
            auto binfo = oper_combine_opB(cindex_c2, cindex_r, ifkr);
            for(const auto& pr : binfo){
               int index = pr.first;
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               int iformula = pr.second;
               int iproc = distribute2('B',ifkr,size,index,int2e.sorb);
               if(iproc == rank){
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  auto Qlc1 = symbolic_compxwf_opQ_su2<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index);
                  auto Bc2r = symbolic_normxwf_opB_su2<Tm>("c2", "r", index, iformula);
                  auto Qlc1_Bc2r = Qlc1.outer_product(Bc2r);
                  double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
                  Qlc1_Bc2r.scale(fac);
                  Qlc1_Bc2r.append_ispins(std::make_tuple(ts,ts,0));
                  formulae.join(Qlc1_Bc2r);
                  counter["QB"] += Qlc1_Bc2r.size();
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Qlc1_Bc2r.display("Qlc1_Bc2r["+std::to_string(index)+"]", print_level);
                  }
               } // iproc
            }

         } // ifNC

         return formulae;
      }

   // primitive form (without factorization)
   template <typename Tm>
      symbolic_task<Tm> symbolic_formulae_twodot(const opersu2_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool ifdistc,
            const bool debug=false){
         auto t0 = tools::get_time();
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         const auto& cindex_l = lqops.cindex;
         const auto& cindex_r = rqops.cindex;
         const auto& cindex_c1 = c1qops.cindex;
         const auto& cindex_c2 = c2qops.cindex;
         int slc1 = cindex_l.size() + cindex_c1.size();
         int sc2r = cindex_c2.size() + cindex_r.size();
         const bool ifNC = (slc1 <= sc2r);
         const bool ifkr = lqops.ifkr; 
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_twodot(su2)"
                  << " mpisize=" << size
                  << " fname=" << fname 
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "cnts::symbolic_formulae_twodot(su2)"
               << " ifkr=" << ifkr
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }
         // generation of Hx
         std::map<std::string,int> counter;
         auto formulae = gen_formulae_twodot_su2(cindex_l,cindex_r,cindex_c1,cindex_c2,ifkr,
               int2e,size,rank,ifdist1,ifdistc,ifsave,counter);
         // reorder if necessary
         if(sort_formulae){
            std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
               {"r",rqops.qket.get_dimAll()},
               {"c1",c1qops.qket.get_dimAll()},
               {"c2",c2qops.qket.get_dimAll()}};
            formulae.sort(dims);
         }
         if(ifsave){
            std::cout << "\nSUMMARY size=" << formulae.size();
            if(ifNC){
               std::cout << " CS:" << counter["CS"] << " SC:" << counter["SC"]
                  << " AP:" << counter["AP"] << " BQ:" << counter["BQ"]
                  << std::endl;
            }else{
               std::cout << " SC:" << counter["SC"] << " CS:" << counter["CS"]
                  << " PA:" << counter["PA"] << " QB:" << counter["QB"]
                  << std::endl;
            }
            formulae.display("total");
            std::cout.rdbuf(backup); // restore cout's original streambuf
            file.close();
         }
         if(rank == 0 and debug){
            auto t1 = tools::get_time();
            int size = formulae.size();
            tools::timing("symbolic_formulae_twodot(su2) with size="+std::to_string(size), t0, t1);
         }
         return formulae;
      }

} // ctns

#endif
