#ifndef SYMBOLIC_FORMULAE_ONEDOT_H
#define SYMBOLIC_FORMULAE_ONEDOT_H

#include "../core/tools.h"
#include "oper_dict.h"
#include "symbolic_task.h"
#include "symbolic_normxwf.h"
#include "symbolic_compxwf.h"
#include "sadmrg/symbolic_formulae_onedot_su2.h"

namespace ctns{

   template <typename Tm>
      symbolic_task<Tm> gen_formulae_onedot(const std::string oplist_l,
            const std::string oplist_r,
            const std::string oplist_c,
            const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c,
            const int isym,
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            std::map<std::string,int>& counter){
         const int print_level = 1;
         const bool ifNC = determine_NCorCN_opH(oplist_l, oplist_r, cindex_l.size(), cindex_r.size()); 
         const auto& cindex = ifNC? cindex_l : cindex_r;
         auto aindex_dist = oper_index_opA_dist(cindex, ifkr, size, rank, int2e.sorb);
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, int2e.sorb);

         symbolic_task<Tm> formulae;
         int idx = 0;

         if(ifNC){
            
            // partition = l|cr
            // 1. H^l 
            counter["H1"] = 0;
            if(!ifdist1 or rank==0){ 
               const double scale = ifkr? 0.25 : 0.5;
               auto Hl = symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale);
               auto Icr = symbolic_prod<Tm>(symbolic_oper("c",'I',0),symbolic_oper("r",'I',0));
               auto Hl_Icr = Hl.product(Icr);
               formulae.append(Hl_Icr);
               counter["H1"] = 1;
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  formulae.display("Hl_Icr", print_level);
               }
            }
            // 2. H^cr
            auto Hcr = symbolic_compxwf_opH<Tm>(oplist_c, oplist_r, "c", "r", cindex_c, cindex_r, 
                  ifkr, int2e.sorb, size, rank, ifdist1);
            counter["H2"] = Hcr.size();
            if(Hcr.size() > 0){
               auto Il = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'I',0)));
               auto Il_Hcr = Il.outer_product(Hcr);
               formulae.join(Il_Hcr);
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  formulae.display("Il_Hcr", print_level);
               }
            }
            // One-index terms:
            // 3. p1^l+*Sp1^cr + h.c.
            counter["CS"] = 0;
            for(const auto& index : cindex_l){
               auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
               auto Scr = symbolic_compxwf_opS<Tm>(oplist_c, oplist_r, "c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
               if(Scr.size() == 0) continue;
               auto Cl_Scr = Cl.outer_product(Scr);
               formulae.join(Cl_Scr);
               counter["CS"] += Cl_Scr.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Cl_Scr.display("Cl_Scr["+std::to_string(index)+"]", print_level);
               }
            }
            // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
            counter["SC"] = 0;
            auto infoC = oper_combine_opC(cindex_c, cindex_r);
            for(const auto& pr : infoC){
               int index = pr.first;
               int iproc = distribute1(ifkr,size,index);
               if(!ifdist1 or iproc==rank){ 
                  int iformula = pr.second;
                  auto Sl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'S',index)));
                  auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
                  Ccr.scale(-1.0);
                  auto Sl_Ccr = Sl.outer_product(Ccr);
                  formulae.join(Sl_Ccr);
                  counter["SC"] += Sl_Ccr.size();
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Sl_Ccr.display("Sl_Ccr["+std::to_string(index)+"]", print_level);
                  }
               }
            }
            // 5. Apq^l*Ppq^cr + h.c.
            counter["AP"] = 0;
            for(const auto& index : aindex_dist){
               auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
               auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr);
               const double wt = ifkr? wfacAP(index) : 1.0;
               Pcr.scale(wt);
               auto Al_Pcr = Al.outer_product(Pcr);
               formulae.join(Al_Pcr);
               counter["AP"] += Al_Pcr.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Al_Pcr.display("Al_Pcr["+std::to_string(index)+"]", print_level);
               }
            }
            // 6. Bps^l*Qps^cr (using Hermicity)
            counter["BQ"] = 0;
            for(const auto& index : bindex_dist){
               auto Bl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'B',index)));
               auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr);
               const double wt = ifkr? wfacBQ(index) : wfac(index);
               Qcr.scale(wt);
               auto Bl_Qcr = Bl.outer_product(Qcr);
               formulae.join(Bl_Qcr);
               counter["BQ"] += Bl_Qcr.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Bl_Qcr.display("Bl_Qcr["+std::to_string(index)+"]", print_level);
               }
            }

         }else{
            
            // partition = lc|r
            // 1. H^lc 
            auto Hlc = symbolic_compxwf_opH<Tm>(oplist_l, oplist_c, "l", "c", cindex_l, cindex_c, 
                  ifkr, int2e.sorb, size, rank, ifdist1);
            counter["H1"] = Hlc.size();
            if(Hlc.size() > 0){
               auto Ir = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'I',0)));
               auto Hlc_Ir = Hlc.outer_product(Ir);
               formulae.join(Hlc_Ir);
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  formulae.display("Hlc_Ir", print_level);
               }
            }
            // 2. H^r
            counter["H2"] = 0;
            if(!ifdist1 or rank==0){ 
               const double scale = ifkr? 0.25 : 0.5;
               auto Ilc = symbolic_prod<Tm>(symbolic_oper("l",'I',0),symbolic_oper("c",'I',0));
               auto Hr = symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale);
               auto Ilc_Hr = Ilc.product(Hr);
               formulae.append(Ilc_Hr);
               counter["H2"] = 1;
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  formulae.display("Ilc_Hr", print_level);
               }
            }
            // One-index terms:
            // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
            counter["SC"] = 0;
            for(const auto& index : cindex_r){
               auto Slc = symbolic_compxwf_opS<Tm>(oplist_l, oplist_c, "l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
               if(Slc.size() == 0) continue;
               Slc.scale(-1.0);
               auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
               auto Slc_Cr = Slc.outer_product(Cr);
               formulae.join(Slc_Cr);
               counter["SC"] += Slc_Cr.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Slc_Cr.display("Slc_Cr["+std::to_string(index)+"]", print_level);
               }
            }
            // 4. p1^lc+*Sp1^r + h.c.
            counter["CS"] = 0;
            auto infoC = oper_combine_opC(cindex_l, cindex_c);
            for(const auto& pr : infoC){
               int index = pr.first;
               int iproc = distribute1(ifkr,size,index);
               if(!ifdist1 or iproc==rank){ 
                  int iformula = pr.second;
                  auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
                  auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
                  auto Clc_Sr = Clc.outer_product(Sr);
                  formulae.join(Clc_Sr);
                  counter["CS"] += Clc_Sr.size();
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Clc_Sr.display("Clc_Sr["+std::to_string(index)+"]", print_level);
                  }
               }
            }
            // 5. Ars^r*Prs^lc + h.c.
            counter["PA"] = 0;
            for(const auto& index : aindex_dist){
               auto Plc = symbolic_compxwf_opP<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr);
               auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
               const double wt = ifkr? wfacAP(index) : 1.0;
               Plc.scale(wt);
               auto Plc_Ar = Plc.outer_product(Ar);
               formulae.join(Plc_Ar);
               counter["PA"] += Plc_Ar.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Plc_Ar.display("Plc_Ar["+std::to_string(index)+"]", print_level);
               }
            }
            // 6. Qqr^lc*Bqr^r (using Hermicity)
            counter["QB"] = 0;
            for(const auto& index : bindex_dist){
               auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr);
               auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
               const double wt = ifkr? wfacBQ(index) : wfac(index);
               Qlc.scale(wt);
               auto Qlc_Br = Qlc.outer_product(Br);
               formulae.join(Qlc_Br);
               counter["QB"] += Qlc_Br.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Qlc_Br.display("Qlc_Br["+std::to_string(index)+"]", print_level);
               }
            }
         
         } // ifNC
         return formulae;
      }

   // primitive form (without factorization)
   template <bool ifab, typename Tm>
      symbolic_task<Tm> symbolic_formulae_onedot(const qoper_dictmap<ifab,Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool ifdistc,
            const bool debug=false){
         auto t0 = tools::get_time();
         const int print_level = 1;
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         const auto& cindex_l = lqops.cindex;
         const auto& cindex_r = rqops.cindex;
         const auto& cindex_c = cqops.cindex;
         const bool ifNC = determine_NCorCN_opH(lqops.oplist, rqops.oplist, cindex_l.size(), cindex_r.size()); 
         const int isym = lqops.isym;
         const bool ifkr = lqops.ifkr;
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_onedot"
                  << " ifab=" << ifab
                  << " mpisize=" << size
                  << " fname=" << fname 
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "ctns::symbolic_formulae_onedot"
               << " isym=" << isym
               << " ifkr=" << ifkr
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }
         // generation of Hx
         std::map<std::string,int> counter;
         symbolic_task<Tm> formulae;
         if(ifab){
            formulae = gen_formulae_onedot(lqops.oplist,rqops.oplist,cqops.oplist,
                  cindex_l,cindex_r,cindex_c,isym,ifkr,
                  int2e,size,rank,ifdist1,ifdistc,ifsave,counter);
         }else{
            formulae = gen_formulae_onedot_su2(lqops.oplist,rqops.oplist,cqops.oplist,
                  cindex_l,cindex_r,cindex_c,isym,ifkr,
                  int2e,size,rank,ifdist1,ifdistc,ifsave,counter);
         }
         // reorder if necessary
         if(sort_formulae){
            std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
               {"r",rqops.qket.get_dimAll()},
               {"c",cqops.qket.get_dimAll()}};
            formulae.sort(dims);
         }
         if(ifsave){
            std::cout << "\nSUMMARY size=" << formulae.size();
            if(ifNC){
               std::cout << " CSnc:" << counter["CS"] << " SCnc:" << counter["SC"]
                  << " APnc:" << counter["AP"] << " BQnc:" << counter["BQ"]
                  << std::endl;
            }else{
               std::cout << " SCcn:" << counter["SC"] << " CScn:" << counter["CS"]
                  << " PAcn:" << counter["PA"] << " QBcn:" << counter["QB"]
                  << std::endl;
            }
            formulae.display("total");
            std::cout.rdbuf(backup); // restore cout's original streambuf
            file.close();
         }
         if(rank == 0 and debug){
            auto t1 = tools::get_time();
            int size = formulae.size();
            tools::timing("symbolic_formulae_onedot with size="+std::to_string(size), t0, t1);
         }
         return formulae;
      }

   // bipartite form (with factorization)
   template <typename Tm>
      bipart_task<Tm> symbolic_formulae_onedot2(const opersu2_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool ifdistc,
            const bool debug=false){
         std::cout << "error: no implementation of symbolic_formulae_onedot2 for su2" << std::endl;
         exit(1);
      }
   template <typename Tm>
      bipart_task<Tm> symbolic_formulae_onedot2(const oper_dictmap<Tm>& qops_dict,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const std::string fname,
            const bool sort_formulae,
            const bool ifdist1,
            const bool ifdistc,
            const bool debug=false){
         auto t0 = tools::get_time();
         const int print_level = 1;
         const auto& lqops = qops_dict.at("l");
         const auto& rqops = qops_dict.at("r");
         const auto& cqops = qops_dict.at("c");
         const auto& cindex_l = lqops.cindex;
         const auto& cindex_r = rqops.cindex;
         const auto& cindex_c = cqops.cindex;
         const int isym = lqops.isym;
         const bool ifkr = lqops.ifkr;
         const bool ifNC = determine_NCorCN_opH(lqops.oplist, rqops.oplist, cindex_l.size(), cindex_r.size()); 
         std::streambuf *psbuf, *backup;
         std::ofstream file;
         bool ifsave = !fname.empty();
         if(ifsave){
            if(rank == 0 and debug){
               std::cout << "ctns::symbolic_formulae_onedot2"
                  << " mpisize=" << size
                  << " fname=" << fname 
                  << std::endl;
            }
            // http://www.cplusplus.com/reference/ios/ios/rdbuf/
            file.open(fname);
            backup = std::cout.rdbuf(); // back up cout's streambuf
            psbuf = file.rdbuf(); // get file's streambuf
            std::cout.rdbuf(psbuf); // assign streambuf to cout
            std::cout << "ctns::symbolic_formulae_onedot2"
               << " isym=" << isym
               << " ifkr=" << ifkr
               << " ifNC=" << ifNC
               << " mpisize=" << size
               << " mpirank=" << rank 
               << std::endl;
         }
         const auto& cindex = ifNC? cindex_l : cindex_r;
         auto aindex_dist = oper_index_opA_dist(cindex, ifkr, size, rank, int2e.sorb);
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, size, rank, int2e.sorb);

         bipart_task<Tm> formulae;
         int idx = 0;
         std::map<std::string,int> counter;

         if(ifNC){
            // partition = l|cr
            // 1. H^l 
            counter["H1"] = 0;
            if(!ifdist1 or rank==0){ 
               const double scale = ifkr? 0.25 : 0.5;
               auto Hl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale));
               auto Hl_Ir = bipart_oper('l',Hl,"Hl_Ir");
               assert(Hl_Ir.parity == 0);
               formulae.push_back(Hl_Ir);
               counter["H1"] = 1;
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  Hl_Ir.display(print_level);
               }
            }
            // 2. H^cr
            auto Hcr = symbolic_compxwf_opH<Tm>(cqops.oplist, rqops.oplist, "c", "r", cindex_c, cindex_r, 
                  ifkr, int2e.sorb, size, rank, ifdist1);
            counter["H2"] = (Hcr.size()>0)? 1 : 0;
            if(Hcr.size() > 0){
               auto Il_Hcr = bipart_oper('r',Hcr,"Il_Hcr");
               assert(Il_Hcr.parity == 0);
               formulae.push_back(Il_Hcr);
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  Il_Hcr.display(print_level);
               }
            }
            // One-index terms:
            // 3. p1^l+*Sp1^cr + h.c.
            counter["CS"] = 0;
            for(const auto& index : cindex_l){
               auto Cl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'C',index)));
               auto Scr = symbolic_compxwf_opS<Tm>(cqops.oplist, rqops.oplist, "c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
               if(Scr.size() == 0) continue;
               auto Cl_Scr = bipart_oper(Cl,Scr,"Cl_Scr["+std::to_string(index)+"]");
               assert(Cl_Scr.parity == 0);
               formulae.push_back(Cl_Scr);
               counter["CS"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Cl_Scr.display(print_level);
               }
            }
            // 4. q2^cr+*Sq2^l + h.c. = -Sq2^l*q2^cr + h.c.
            counter["SC"] = 0;
            auto infoC = oper_combine_opC(cindex_c, cindex_r);
            for(const auto& pr : infoC){
               int index = pr.first;
               int iproc = distribute1(ifkr,size,index);
               if(!ifdist1 or iproc==rank){ 
                  int iformula = pr.second;
                  auto Sl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'S',index)));
                  auto Ccr = symbolic_normxwf_opC<Tm>("c", "r", index, iformula);
                  Ccr.scale(-1.0);
                  auto Sl_Ccr = bipart_oper(Sl,Ccr,"Sl_Ccr["+std::to_string(index)+"]");
                  assert(Sl_Ccr.parity == 0);
                  formulae.push_back(Sl_Ccr);
                  counter["SC"] += 1;
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Sl_Ccr.display(print_level);
                  }
               }
            }
            // 5. Apq^l*Ppq^cr + h.c.
            counter["AP"] = 0;
            for(const auto& index : aindex_dist){
               auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
               auto Pcr = symbolic_compxwf_opP<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr);
               const double wt = ifkr? wfacAP(index) : 1.0;
               Pcr.scale(wt);
               auto Al_Pcr = bipart_oper(Al,Pcr,"Al_Pcr["+std::to_string(index)+"]");
               assert(Al_Pcr.parity == 0);
               formulae.push_back(Al_Pcr);
               counter["AP"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Al_Pcr.display(print_level);
               }
            }
            // 6. Bps^l*Qps^cr (using Hermicity)
            counter["BQ"] = 0;
            for(const auto& index : bindex_dist){
               auto Bl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'B',index)));
               auto Qcr = symbolic_compxwf_opQ<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index, isym, ifkr);
               const double wt = ifkr? wfacBQ(index) : wfac(index);
               Qcr.scale(wt);
               auto Bl_Qcr = bipart_oper(Bl,Qcr,"Bl_Qcr["+std::to_string(index)+"]");
               assert(Bl_Qcr.parity == 0);
               formulae.push_back(Bl_Qcr);
               counter["BQ"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Bl_Qcr.display(print_level);
               }
            }
         }else{
            // partition = lc|r
            // 1. H^lc 
            auto Hlc = symbolic_compxwf_opH<Tm>(lqops.oplist, cqops.oplist, "l", "c", cindex_l, cindex_c, 
                  ifkr, int2e.sorb, size, rank, ifdist1);
            counter["H1"] = (Hlc.size()>0)? 1 : 0;
            if(Hlc.size() > 0){
               auto Hlc_Ir = bipart_oper('l',Hlc,"Hlc_Ir");
               assert(Hlc_Ir.parity == 0);
               formulae.push_back(Hlc_Ir);
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  Hlc_Ir.display(print_level);
               }
            }
            // 2. H^r
            counter["H2"] = 0;
            if(!ifdist1 or rank==0){ 
               const double scale = ifkr? 0.25 : 0.5;
               auto Hr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale));
               auto Ilc_Hr = bipart_oper('r',Hr,"Ilc_Hr");
               assert(Ilc_Hr.parity == 0);
               formulae.push_back(Ilc_Hr);
               counter["H2"] = 1;
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  Ilc_Hr.display(print_level);
               }
            }
            // One-index terms:
            // 3. q2^r+*Sq2^lc + h.c. = -Sq2^lc*q2^r + h.c.
            counter["SC"] = 0;
            for(const auto& index : cindex_r){
               auto Slc = symbolic_compxwf_opS<Tm>(lqops.oplist, cqops.oplist, "l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr, size, rank, ifdist1, ifdistc);
               if(Slc.size() == 0) continue;
               Slc.scale(-1.0);
               auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
               auto Slc_Cr = bipart_oper(Slc,Cr,"Slc_Cr["+std::to_string(index)+"]");
               assert(Slc_Cr.parity == 0);
               formulae.push_back(Slc_Cr);
               counter["SC"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Slc_Cr.display(print_level);
               }
            }
            // 4. p1^lc+*Sp1^r + h.c.
            counter["CS"] = 0;
            auto infoC = oper_combine_opC(cindex_l, cindex_c);
            for(const auto& pr : infoC){
               int index = pr.first;
               int iproc = distribute1(ifkr,size,index);
               if(!ifdist1 or iproc==rank){ 
                  int iformula = pr.second;
                  auto Clc = symbolic_normxwf_opC<Tm>("l", "c", index, iformula);
                  auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
                  auto Clc_Sr = bipart_oper(Clc,Sr,"Clc_Sr["+std::to_string(index)+"]");
                  assert(Clc_Sr.parity == 0);
                  formulae.push_back(Clc_Sr);
                  counter["CS"] += 1;
                  if(ifsave){ 
                     std::cout << "idx=" << idx++;
                     Clc_Sr.display(print_level);
                  }
               }
            }
            // 5. Ars^r*Prs^lc + h.c.
            counter["PA"] = 0;
            for(const auto& index : aindex_dist){
               auto Plc = symbolic_compxwf_opP<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr);
               auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
               const double wt = ifkr? wfacAP(index) : 1.0;
               Plc.scale(wt);
               auto Plc_Ar = bipart_oper(Plc,Ar,"Plc_Ar["+std::to_string(index)+"]");
               assert(Plc_Ar.parity == 0);
               formulae.push_back(Plc_Ar);
               counter["PA"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Plc_Ar.display(print_level);
               }
            }
            // 6. Qqr^lc*Bqr^r (using Hermicity)
            counter["QB"] = 0;
            for(const auto& index : bindex_dist){
               auto Qlc = symbolic_compxwf_opQ<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index, isym, ifkr);
               auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
               const double wt = ifkr? wfacBQ(index) : wfac(index);
               Qlc.scale(wt);
               auto Qlc_Br = bipart_oper(Qlc,Br,"Qlc_Br["+std::to_string(index)+"]");
               assert(Qlc_Br.parity == 0);
               formulae.push_back(Qlc_Br);
               counter["QB"] += 1;
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Qlc_Br.display(print_level);
               }
            }
         } // ifNC

         if(sort_formulae){
            std::map<std::string,int> dims = {{"l",lqops.qket.get_dimAll()},
               {"r",rqops.qket.get_dimAll()},
               {"c",cqops.qket.get_dimAll()}};
            sort(formulae, dims);
         }
         if(ifsave){
            std::cout << "\nSUMMARY size=" << idx;
            if(ifNC){
               std::cout << " CSnc:" << counter["CS"] << " SCnc:" << counter["SC"]
                  << " APnc:" << counter["AP"] << " BQnc:" << counter["BQ"]
                  << std::endl;
            }else{
               std::cout << " SCcn:" << counter["SC"] << " CScn:" << counter["CS"]
                  << " PAcn:" << counter["PA"] << " QBcn:" << counter["QB"]
                  << std::endl;
            }
            display(formulae, "total");
            std::cout.rdbuf(backup); // restore cout's original streambuf
            file.close();
         }
         if(rank == 0 and debug){
            auto t1 = tools::get_time();
            int size = formulae.size();
            tools::timing("symbolic_formulae_onedot2 with size="+std::to_string(size), t0, t1);
         }
         return formulae;
      }

} // ctns

#endif
