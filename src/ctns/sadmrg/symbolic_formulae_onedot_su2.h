#ifndef SYMBOLIC_FORMULAE_ONEDOT_SU2_H
#define SYMBOLIC_FORMULAE_ONEDOT_SU2_H

#include "../../core/tools.h"
#include "../oper_dict.h"
#include "../symbolic_oper.h"
#include "symbolic_normxwf_su2.h"
#include "symbolic_compxwf_su2.h"

namespace ctns{

   template <typename Tm>
      symbolic_task<Tm> gen_formulae_onedot_su2(const std::string oplist_l,
            const std::string oplist_r,
            const std::string oplist_c,
            const std::vector<int>& cindex_l,
            const std::vector<int>& cindex_r,
            const std::vector<int>& cindex_c,
            const int isym, // for uniform interface only
            const bool ifkr,
            const integral::two_body<Tm>& int2e,
            const int& size,
            const int& rank,
            const bool ifdist1,
            const bool ifdistc,
            const bool ifsave,
            std::map<std::string,int>& counter){
         assert(isym == 3);
         const int print_level = 1;
         const bool ifNC = determine_NCorCN_Ham(oplist_l, oplist_r, cindex_l.size(), cindex_r.size()); 
         const auto& cindex = ifNC? cindex_l : cindex_r;
         auto aindex_dist = oper_index_opA_dist(cindex, ifkr, isym, size, rank, int2e.sorb);
         auto bindex_dist = oper_index_opB_dist(cindex, ifkr, isym, size, rank, int2e.sorb);

         symbolic_task<Tm> formulae;
         size_t idx = 0;

         if(ifNC){
            
            // partition = l|cr
            // 1. H^l*Icr 
            counter["H1"] = 0;
            if(!ifdist1 or rank==0){
               const double scale = 0.5;
               auto Hl = symbolic_prod<Tm>(symbolic_oper("l",'H',0), scale);
               auto Icr = symbolic_prod<Tm>(symbolic_oper("c",'I',0),symbolic_oper("r",'I',0));
               Icr.ispins.push_back(std::make_tuple(0,0,0));
               auto Hl_Icr = Hl.product(Icr);
               Hl_Icr.ispins.push_back(std::make_tuple(0,0,0));
               formulae.append(Hl_Icr);
               counter["H1"] = 1;
               if(ifsave){
                  std::cout << "idx=" << idx++;
                  formulae.display("Hl_Icr", print_level);
               }
            }
            // 2. Il*H^cr
            auto Hcr = symbolic_compxwf_opH_su2<Tm>(oplist_c, oplist_r, "c", "r", cindex_c, cindex_r, 
                  int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
            counter["H2"] = Hcr.size();
            if(Hcr.size() > 0){
               auto Il = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'I',0)));
               auto Il_Hcr = Il.outer_product(Hcr);
               Il_Hcr.append_ispins(std::make_tuple(0,0,0));
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
               auto Scr = symbolic_compxwf_opS_su2<Tm>(oplist_c, oplist_r, "c", "r", cindex_c, cindex_r,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Scr.size() == 0) continue;
               auto Cl_Scr = Cl.outer_product(Scr);
               Cl_Scr.scale(std::sqrt(2.0));
               Cl_Scr.append_ispins(std::make_tuple(1,1,0));
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
                  auto Ccr = symbolic_normxwf_opC_su2<Tm>("c", "r", index, iformula);
                  auto Sl_Ccr = Sl.outer_product(Ccr);
                  Sl_Ccr.scale(std::sqrt(2.0));
                  Sl_Ccr.append_ispins(std::make_tuple(1,1,0));
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
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2;
               auto Al = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'A',index)));
               auto Pcr = symbolic_compxwf_opP_su2<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index);
               auto Al_Pcr = Al.outer_product(Pcr);
               double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
               Al_Pcr.scale(fac);
               Al_Pcr.append_ispins(std::make_tuple(ts,ts,0));
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
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               auto Bl = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("l",'B',index)));
               auto Qcr = symbolic_compxwf_opQ_su2<Tm>("c", "r", cindex_c, cindex_r,
                     int2e, index);
               auto Bl_Qcr = Bl.outer_product(Qcr);
               double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
               Bl_Qcr.scale(fac);
               Bl_Qcr.append_ispins(std::make_tuple(ts,ts,0)); 
               formulae.join(Bl_Qcr);
               counter["BQ"] += Bl_Qcr.size();
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  Bl_Qcr.display("Bl_Qcr["+std::to_string(index)+"]", print_level);
               }
            }

         }else{
            
            // partition = lc|r
            // 1. H^lc*Ir 
            auto Hlc = symbolic_compxwf_opH_su2<Tm>(oplist_l, oplist_c, "l", "c", cindex_l, cindex_c, 
                  int2e, ifkr, int2e.sorb, size, rank, ifdist1, ifdistc);
            counter["H1"] = Hlc.size();
            if(Hlc.size() > 0){
               auto Ir = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'I',0)));
               auto Hlc_Ir = Hlc.outer_product(Ir);
               Hlc_Ir.append_ispins(std::make_tuple(0,0,0));
               formulae.join(Hlc_Ir);
               if(ifsave){ 
                  std::cout << "idx=" << idx++;
                  formulae.display("Hlc_Ir", print_level);
               }
            }
            // 2. Ilc*H^r
            counter["H2"] = 0;
            if(!ifdist1 or rank==0){ 
               const double scale = 0.5;
               auto Ilc = symbolic_prod<Tm>(symbolic_oper("l",'I',0),symbolic_oper("c",'I',0));
               Ilc.ispins.push_back(std::make_tuple(0,0,0));
               auto Hr = symbolic_prod<Tm>(symbolic_oper("r",'H',0), scale);
               auto Ilc_Hr = Ilc.product(Hr);
               Ilc_Hr.ispins.push_back(std::make_tuple(0,0,0));
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
               auto Slc = symbolic_compxwf_opS_su2<Tm>(oplist_l, oplist_c, "l", "c", cindex_l, cindex_c,
                     int2e, index, ifkr, size, rank, ifdist1, ifdistc);
               if(Slc.size() == 0) continue;
               auto Cr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'C',index)));
               auto Slc_Cr = Slc.outer_product(Cr);
               Slc_Cr.scale(std::sqrt(2.0));
               Slc_Cr.append_ispins(std::make_tuple(1,1,0));
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
                  auto Clc = symbolic_normxwf_opC_su2<Tm>("l", "c", index, iformula);
                  auto Sr = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'S',index)));
                  auto Clc_Sr = Clc.outer_product(Sr);
                  Clc_Sr.scale(std::sqrt(2.0));
                  Clc_Sr.append_ispins(std::make_tuple(1,1,0));
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
               auto pq = oper_unpack(index);
               int p = pq.first, kp = p/2, sp = p%2;
               int q = pq.second, kq = q/2, sq = q%2;
               int ts = (sp!=sq)? 0 : 2; 
               auto Plc = symbolic_compxwf_opP_su2<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index);
               auto Ar = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'A',index)));
               auto Plc_Ar = Plc.outer_product(Ar);
               double fac = (ts==0)? ((kp==kq)? -0.5 : -1.0) : std::sqrt(3.0);
               Plc_Ar.scale(fac);
               Plc_Ar.append_ispins(std::make_tuple(ts,ts,0));
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
               auto ps = oper_unpack(index);
               int p = ps.first, kp = p/2, sp = p%2;
               int s = ps.second, ks = s/2, ss = s%2;
               int ts = (sp!=ss)? 2 : 0;
               auto Qlc = symbolic_compxwf_opQ_su2<Tm>("l", "c", cindex_l, cindex_c,
                     int2e, index);
               auto Br = symbolic_task<Tm>(symbolic_prod<Tm>(symbolic_oper("r",'B',index)));
               auto Qlc_Br = Qlc.outer_product(Br);
               double fac = ((kp==ks)? 0.5 : 1.0)*((ts==0)? 1.0 : -std::sqrt(3.0));
               Qlc_Br.scale(fac);
               Qlc_Br.append_ispins(std::make_tuple(ts,ts,0));
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

} // ctns

#endif
