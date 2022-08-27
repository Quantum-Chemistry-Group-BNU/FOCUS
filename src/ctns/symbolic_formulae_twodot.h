#ifndef SYMBOLIC_FORMULAE_TWODOT_H
#define SYMBOLIC_FORMULAE_TWODOT_H

#include "../core/tools.h"
#include "oper_dict.h"
#include "symbolic_oper.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"

namespace ctns{

   template <typename Tm>
      symbolic_task<Tm> gen_formulae_twodot(const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c1,
            const std::vector<int>& cindex_c2,
            const int isym,
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
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
         auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", cindex_l, cindex_c1, 
               ifkr, size, rank, ifdist1);
         counter["H1"] = Hlc1.size();
         if(Hlc1.size() > 0){
            formulae.join(Hlc1);
            if(ifsave){ 
               std::cout << "idx=" << idx++; 
               Hlc1.display("Hlc1", print_level);
            }
         }
         // H[c2r]
         auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", cindex_c2, cindex_r, 
               ifkr, size, rank, ifdist1);
         counter["H2"] = Hc2r.size();
         if(Hc2r.size() > 0){
            formulae.join(Hc2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Hc2r.display("Hc2r", print_level);
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
            auto Clc1 = symbolic_normxwf_opC<Tm>("l", "c1", index, iformula);
            auto Sc2r = symbolic_compxwf_opS<Tm>("c2", "r", cindex_c2, cindex_r,
                  int2e, index, isym, ifkr, size, rank, ifdist1);
            if(Sc2r.size() == 0) continue;
            auto Clc1_Sc2r = Clc1.outer_product(Sc2r);
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
            auto Slc1 = symbolic_compxwf_opS<Tm>("l", "c1", cindex_l, cindex_c1,
                  int2e, index, isym, ifkr, size, rank, ifdist1);
            if(Slc1.size() == 0) continue;
            auto Cc2r = symbolic_normxwf_opC<Tm>("c2", "r", index, iformula);
            Cc2r.scale(-1.0);
            auto Slc1_Cc2r = Slc1.outer_product(Cc2r);
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
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula, ifkr);
                  auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index, isym, ifkr);
                  const double wt = ifkr? wfacAP(index) : 1.0;
                  Pc2r.scale(wt);
                  auto Alc1_Pc2r = Alc1.outer_product(Pc2r);
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
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula, ifkr);
                  auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index, isym, ifkr);
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  const double wt = ifkr? wfacBQ(index) : wfac(index);
                  Qc2r.scale(wt);
                  auto Blc1_Qc2r = Blc1.outer_product(Qc2r);
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
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Plc1 = symbolic_compxwf_opP<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index, isym, ifkr);
                  auto Ac2r = symbolic_normxwf_opA<Tm>("c2", "r", index, iformula, ifkr);
                  const double wt = ifkr? wfacAP(index) : 1.0;
                  Plc1.scale(wt);
                  auto Plc1_Ac2r = Plc1.outer_product(Ac2r);
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
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  auto Qlc1 = symbolic_compxwf_opQ<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index, isym, ifkr);
                  auto Bc2r = symbolic_normxwf_opB<Tm>("c2", "r", index, iformula, ifkr);
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  const double wt = ifkr? wfacBQ(index) : wfac(index);
                  Qlc1.scale(wt);
                  auto Qlc1_Bc2r = Qlc1.outer_product(Bc2r);
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
      symbolic_task<Tm> symbolic_formulae_twodot(const oper_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
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
         const int isym = lqops.isym;
         const bool ifkr = lqops.ifkr; 
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_twodot"
                  << " mpisize=" << size
                  << " fname=" << fname 
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "cnts::symbolic_formulae_twodot"
               << " isym=" << isym
               << " ifkr=" << ifkr
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }
         // generation of Hx
         std::map<std::string,int> counter;
         auto formulae = gen_formulae_twodot(cindex_l,cindex_r,cindex_c1,cindex_c2,isym,ifkr,
               int2e,size,rank,ifdist1,ifsave,counter);
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
            tools::timing("symbolic_formulae_twodot with size="+std::to_string(size), t0, t1);
         }
         return formulae;
      }

   // bipartite form (with factorization)
   template <typename Tm>
      bipart_task<Tm> symbolic_formulae_twodot2(const oper_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool debug=false){
         auto t0 = tools::get_time();
         const int print_level = 1;
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& c1qops = qops_dict.at("c1");
         const auto& c2qops = qops_dict.at("c2");
         const auto& cindex_l = lqops.cindex;
         const auto& cindex_r = rqops.cindex;
         const auto& cindex_c1 = c1qops.cindex;
         const auto& cindex_c2 = c2qops.cindex;
         const int isym = lqops.isym;
         const bool ifkr = lqops.ifkr;
         int slc1 = cindex_l.size() + cindex_c1.size();
         int sc2r = cindex_c2.size() + cindex_r.size();
         const bool ifNC = (slc1 <= sc2r);
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_twodot2"
                  << " mpisize=" << size
                  << " fname=" << fname 
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "cnts::symbolic_formulae_twodot2"
               << " isym=" << isym
               << " ifkr=" << ifkr
               << " ifNC=" << ifNC
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }

         bipart_task<Tm> formulae;
         int idx = 0;
         std::map<std::string,int> counter;

         // Local terms:
         // H[lc1]
         auto Hlc1 = symbolic_compxwf_opH<Tm>("l", "c1", cindex_l, cindex_c1, 
               ifkr, size, rank, ifdist1);
         counter["H1"] = (Hlc1.size()>0)? 1 : 0;
         if(Hlc1.size() > 0){
            auto Hlc1_Ic2r = bipart_oper('l',Hlc1,"Hlc1_Ic2r");
            assert(Hlc1_Ic2r.parity == 0);
            formulae.push_back(Hlc1_Ic2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++; 
               Hlc1_Ic2r.display(print_level);
            }
         }
         // H[c2r]
         auto Hc2r = symbolic_compxwf_opH<Tm>("c2", "r", cindex_c2, cindex_r, 
               ifkr, size, rank, ifdist1);
         counter["H2"] = (Hc2r.size()>0)? 1 : 0;
         if(Hc2r.size() > 0){
            auto Ilc1_Hc2r = bipart_oper('r',Hc2r,"Ilc1_Hc2r");
            assert(Ilc1_Hc2r.parity == 0);
            formulae.push_back(Ilc1_Hc2r);
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Ilc1_Hc2r.display(print_level);
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
            auto Clc1 = symbolic_normxwf_opC<Tm>("l", "c1", index, iformula);
            auto Sc2r = symbolic_compxwf_opS<Tm>("c2", "r", cindex_c2, cindex_r,
                  int2e, index, isym, ifkr, size, rank, ifdist1);
            if(Sc2r.size() == 0) continue;
            auto Clc1_Sc2r = bipart_oper(Clc1,Sc2r,"Clc1_Sc2r["+std::to_string(index)+"]");
            assert(Clc1_Sc2r.parity == 0);
            formulae.push_back(Clc1_Sc2r);
            counter["CS"] += 1;
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Clc1_Sc2r.display(print_level);
            }
         }
         // 4. sum_q2 q2^+[C2R]*Sq2^[LC1] + h.c. = -Sq2^[LC1]*q2^+[C2R] + h.c.
         counter["SC"] = 0;
         auto infoC2 = oper_combine_opC(cindex_c2, cindex_r);
         for(const auto& pr : infoC2){
            int index = pr.first;
            int iformula = pr.second;
            // q2^C2R+*Sq2^LC1 = -Sq2^LC1*q2^C2R+ & Sq2^LC1+*q2^C2R
            auto Slc1 = symbolic_compxwf_opS<Tm>("l", "c1", cindex_l, cindex_c1,
                  int2e, index, isym, ifkr, size, rank, ifdist1);
            if(Slc1.size() == 0) continue;
            auto Cc2r = symbolic_normxwf_opC<Tm>("c2", "r", index, iformula);
            Cc2r.scale(-1.0);
            auto Slc1_Cc2r = bipart_oper(Slc1,Cc2r,"Slc1_Cc2r["+std::to_string(index)+"]");
            assert(Slc1_Cc2r.parity == 0);
            formulae.push_back(Slc1_Cc2r);
            counter["SC"] += 1;
            if(ifsave){ 
               std::cout << "idx=" << idx++;
               Slc1_Cc2r.display(print_level);
            }
         }

         // Two-index terms:
         if(ifNC){
            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["AP"] = 0;
            auto ainfo = oper_combine_opA(cindex_l, cindex_c1, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Alc1 = symbolic_normxwf_opA<Tm>("l", "c1", index, iformula, ifkr);
                  auto Pc2r = symbolic_compxwf_opP<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index, isym, ifkr);
                  const double wt = ifkr? wfacAP(index) : 1.0;
                  Pc2r.scale(wt);
                  auto Alc1_Pc2r = bipart_oper(Alc1,Pc2r,"Alc1_Pc2r["+std::to_string(index)+"]");
                  assert(Alc1_Pc2r.parity == 0);
                  formulae.push_back(Alc1_Pc2r);
                  counter["AP"] += 1;
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Alc1_Pc2r.display(print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["BQ"] = 0;
            auto binfo = oper_combine_opB(cindex_l, cindex_c1, ifkr);
            for(const auto& pr : binfo){
               int index = pr.first;
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  auto Blc1 = symbolic_normxwf_opB<Tm>("l", "c1", index, iformula, ifkr);
                  auto Qc2r = symbolic_compxwf_opQ<Tm>("c2", "r", cindex_c2, cindex_r,
                        int2e, index, isym, ifkr);
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  const double wt = ifkr? wfacBQ(index) : wfac(index);
                  Qc2r.scale(wt);
                  auto Blc1_Qc2r = bipart_oper(Blc1,Qc2r,"Blc1_Qc2r["+std::to_string(index)+"]");
                  assert(Blc1_Qc2r.parity == 0);
                  formulae.push_back(Blc1_Qc2r);
                  counter["BQ"] += 1;
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Blc1_Qc2r.display(print_level);
                  }
               } // iproc
            }
         }else{
            // 5. Apq^LC1*Ppq^C2R + h.c. or Ars^C2R*Prs^LC1 + h.c.
            counter["PA"] = 0;
            auto ainfo = oper_combine_opA(cindex_c2, cindex_r, ifkr);
            for(const auto& pr : ainfo){
               int index = pr.first;
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  // Apq*Ppq + Apq^+*Ppq^+
                  auto Plc1 = symbolic_compxwf_opP<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index, isym, ifkr);
                  auto Ac2r = symbolic_normxwf_opA<Tm>("c2", "r", index, iformula, ifkr);
                  const double wt = ifkr? wfacAP(index) : 1.0;
                  Plc1.scale(wt);
                  auto Plc1_Ac2r = bipart_oper(Plc1,Ac2r,"Plc1_Ac2r["+std::to_string(index)+"]");
                  assert(Plc1_Ac2r.parity == 0);
                  formulae.push_back(Plc1_Ac2r);
                  counter["PA"] += 1;
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Plc1_Ac2r.display(print_level);
                  }
               } // iproc
            }
            // 6. Bps^LC1*Qps^C2R + h.c. or Qqr^LC1*Bqr^C2R
            counter["QB"] = 0;
            auto binfo = oper_combine_opB(cindex_c2, cindex_r, ifkr);
            for(const auto& pr : binfo){
               int index = pr.first;
               int iformula = pr.second;
               int iproc = distribute2(ifkr,size,index);
               if(iproc == rank){
                  auto Qlc1 = symbolic_compxwf_opQ<Tm>("l", "c1", cindex_l, cindex_c1,
                        int2e, index, isym, ifkr);
                  auto Bc2r = symbolic_normxwf_opB<Tm>("c2", "r", index, iformula, ifkr);
                  // Bpq*Qpq + Bpq^+*Qpq^+
                  const double wt = ifkr? wfacBQ(index) : wfac(index);
                  Qlc1.scale(wt);
                  auto Qlc1_Bc2r = bipart_oper(Qlc1,Bc2r,"Qlc1_Bc2r["+std::to_string(index)+"]");
                  assert(Qlc1_Bc2r.parity == 0);
                  formulae.push_back(Qlc1_Bc2r);
                  counter["QB"] += 1;
                  if(ifsave){
                     std::cout << "idx=" << idx++;
                     Qlc1_Bc2r.display(print_level);
                  }
               } // iproc
            }
         } // ifNC

         if(sort_formulae){
            std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
               {"r",rqops.qket.get_dimAll()},
               {"c1",c1qops.qket.get_dimAll()},
               {"c2",c2qops.qket.get_dimAll()}};
            sort(formulae, dims);
         }
         if(ifsave){
            std::cout << "\nSUMMARY size=" << idx;
            if(ifNC){
               std::cout << " CS:" << counter["CS"] << " SC:" << counter["SC"]
                  << " AP:" << counter["AP"] << " BQ:" << counter["BQ"]
                  << std::endl;
            }else{
               std::cout << " SC:" << counter["SC"] << " CS:" << counter["CS"]
                  << " PA:" << counter["PA"] << " QB:" << counter["QB"]
                  << std::endl;
            }
            display(formulae, "total");
            std::cout.rdbuf(backup); // restore cout's original streambuf
            file.close();
         }
         if(rank == 0 and debug){
            auto t1 = tools::get_time();
            int size = formulae.size();
            tools::timing("symbolic_formulae_twodot2 with size="+std::to_string(size), t0, t1);
         }
         return formulae;
      }

} // ctns

#endif
